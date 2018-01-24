/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.mkl.MklDnn.{EngineType, StreamType}
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.nn.{ErrorInfo, InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Linear[T: ClassTag](
  val inputSize: Int,
  val outputSize: Int,
  val withBias: Boolean = true,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {
  val weight: Tensor[T] =
    if (initWeight != null) initWeight else Tensor[T](outputSize, inputSize)
  val bias: Tensor[T] =
    if (initBias != null) initBias else if (withBias) Tensor[T](outputSize) else null
  val addBuffer: Tensor[T] = Tensor[T]()

  val gradWeight: Tensor[T] =
    if (initGradWeight != null) initGradWeight else Tensor[T]()
  val gradBias: Tensor[T] =
    if (initGradBias != null) initGradBias else if (withBias) Tensor[T]() else null

  val diffWeight: Tensor[T] = Tensor[T].resizeAs(gradWeight)
  val diffBias: Tensor[T] = Tensor[T].resizeAs(gradBias)

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.OUT_IN)
    }
    if (initBias == null) {
      Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    }
    zeroGradParameters()
  }

  @transient var engine = 0L
  @transient var stream = 0L

  @transient var forwardPrim = 0L
  @transient var backDataPrim = 0L
  @transient var backWeightPrim = 0L

  @transient var forwardPrimDesc = 0L

  @transient private var forwardPrimBuffer: ArrayBuffer[Long] = _
  @transient private var backwardDataPrimBuffer: ArrayBuffer[Long] = _
  @transient private var backwardWeightPrimBuffer: ArrayBuffer[Long] = _

  val inputPrim, weightPrim, biasPrim, outputPrim: MemoryPrimitive[T] =
    new MemoryPrimitive[T]()
  val gradInputPrim, gradWeightPrim, gradBiasPrim, gradOutputPrim: MemoryPrimitive[T] =
    new MemoryPrimitive[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    } else if (input.dim() == 2) {
      val nFrame = input.size(1)
      val t = Array(nFrame, weight.size(1))
      output.resize(t)
    } else if (input.dim() == 4) {
      output.resize(input.size(1), weight.size(1))
      weight.resize(weight.size(1), input.size(2), input.size(3), input.size(4))
    }

    if (forwardPrim == 0L) {
      if (engine == 0L) engine = this.getDnnEngine(0)
      if (stream == 0L) stream = this.getStream()

      forwardPrimBuffer = ArrayBuffer.empty[Long]
      val weightMemDesc = if (input.dim() == 4) {
        MklDnn.MemoryDescInit(weight.dim(),
          weight.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(weight.dim(), weight.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }
      val biasMemDesc = MklDnn.MemoryDescInit(bias.dim(), bias.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

      val dstMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(output.dim() + 1, Array(1) ++ output.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(output.dim(), output.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      val srcMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(input.dim() + 1, Array(1) ++ input.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      val format = input.dim() match {
        case 1 => MklDnn.MemoryFormat.nc
        case 2 => MklDnn.MemoryFormat.nc
        case 4 => MklDnn.MemoryFormat.nchw
      }

      val weightFormat = input.dim() match {
        case 1 => MklDnn.MemoryFormat.oi
        case 2 => MklDnn.MemoryFormat.oi
        case 4 => MklDnn.MemoryFormat.oihw
      }

      val opDesc = MklDnn.LinearForwardDescInit(MklDnn.PropKind.forward,
        srcMemDesc, weightMemDesc, biasMemDesc, dstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)
      forwardPrimDesc = opPrimDesc

      inputPrim.initUser(input, MklDnn.DataType.f32, format, engine)
      inputPrim.initInternal(opPrimDesc, MklDnn.Query.src_pd)
      weightPrim.initUser(weight, MklDnn.DataType.f32, weightFormat, engine)
      weightPrim.initInternal(opPrimDesc, MklDnn.Query.weights_pd)
      biasPrim.initUser(bias, MklDnn.DataType.f32, MklDnn.MemoryFormat.x, engine)

      // we create output primitive with any format
      outputPrim.initUser(output, opPrimDesc, MklDnn.Query.dst_pd, 0)

      val srcs = Array(inputPrim.workPrim(), weightPrim.workPrim(), biasPrim.workPrim())
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(outputPrim.workPrim())

      forwardPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)

      if (inputPrim.reorder != 0) {
        forwardPrimBuffer += inputPrim.reorder
      }

      if (weightPrim.reorder != 0) {
        forwardPrimBuffer += weightPrim.reorder
      }

      forwardPrimBuffer += forwardPrim
    }

    inputPrim.setHandle(input)
    weightPrim.setHandle(weight)
    biasPrim.setHandle(bias)
    outputPrim.setHandle(output)

    MklDnn.StreamSubmit(stream, forwardPrimBuffer.length, forwardPrimBuffer.toArray)

    inputPrim.releaseHandle()
    weightPrim.releaseHandle()
    biasPrim.releaseHandle()
    outputPrim.releaseHandle()

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    require(gradOutput.size().deep == output.size().deep,
      s"output size should be the same as gradOutput")

    if (backDataPrim == 0L) {
      backwardDataPrimBuffer = ArrayBuffer.empty[Long]
      val diffSrcMemDesc = if (gradInput.dim() == 1) {
        MklDnn.MemoryDescInit(gradInput.dim() + 1, Array(1) ++ gradInput.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(gradInput.dim(), gradInput.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      val weightMemDesc = if (input.dim() == 4) {
        MklDnn.MemoryDescInit(weight.dim(), weight.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(weight.dim(), weight.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      val diffDstMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(gradOutput.dim() + 1, Array(1) ++ gradOutput.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(output.dim(), output.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      val format = input.dim() match {
        case 1 => MklDnn.MemoryFormat.nc
        case 2 => MklDnn.MemoryFormat.nc
        case 4 => MklDnn.MemoryFormat.nchw
      }

      val weightFormat = input.dim() match {
        case 1 => MklDnn.MemoryFormat.oi
        case 2 => MklDnn.MemoryFormat.oi
        case 4 => MklDnn.MemoryFormat.oihw
      }

      val opDesc = MklDnn.LinearBackwardDataDescInit(diffSrcMemDesc, weightMemDesc,
        diffDstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, forwardPrimDesc)

      gradOutputPrim.initUser(gradOutput, MklDnn.DataType.f32, format, engine)
      gradOutputPrim.initInternal(opPrimDesc, MklDnn.Query.diff_dst_pd)
      gradInputPrim.initUser(gradInput, opPrimDesc, MklDnn.Query.diff_src_pd, 0)

      val srcs = Array(gradOutputPrim.workPrim(), weightPrim.workPrim())
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(gradInputPrim.workPrim())

      backDataPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)

      if (gradOutputPrim.reorder != 0) {
        backwardDataPrimBuffer += gradOutputPrim.reorder
      }

      backwardDataPrimBuffer += backDataPrim
    }

    gradOutputPrim.setHandle(gradOutput)
    weightPrim.setHandle(weight)
    gradInputPrim.setHandle(gradInput)

    MklDnn.StreamSubmit(stream, backwardDataPrimBuffer.length, backwardDataPrimBuffer.toArray)

    gradInputPrim.releaseHandle()
    gradOutputPrim.releaseHandle()
    weightPrim.releaseHandle()

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    gradWeight.resizeAs(weight)
    if (withBias) {
      gradBias.resizeAs(bias)
    }

    diffWeight.resizeAs(weight)
    if (withBias) {
      diffBias.resizeAs(bias)
    }

    if (backWeightPrim == 0) {
      backwardWeightPrimBuffer = ArrayBuffer.empty[Long]
      val diffWeightMemDesc = if (input.dim() == 4) {
        MklDnn.MemoryDescInit(gradWeight.dim(), gradWeight.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(gradWeight.dim(), gradWeight.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      val diffBiasMemDesc = MklDnn.MemoryDescInit(gradBias.dim(), gradBias.size(),
        MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

      val srcMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(input.dim() + 1, Array(1) ++ input.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      val diffDstMemDesc = if (input.dim() == 1) {
        MklDnn.MemoryDescInit(gradOutput.dim() + 1, Array(1) ++ gradOutput.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      } else {
        MklDnn.MemoryDescInit(output.dim(), output.size(),
          MklDnn.DataType.f32, MklDnn.MemoryFormat.any)
      }

      println(inputPrim.workPrim())
      println(gradOutputPrim.workPrim())
      val opDesc = MklDnn.LinearBackwardWeightsDescInit(
        srcMemDesc, diffWeightMemDesc, diffBiasMemDesc, diffDstMemDesc)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, forwardPrimDesc)

      val weightFormat = input.dim() match {
        case 1 => MklDnn.MemoryFormat.oi
        case 2 => MklDnn.MemoryFormat.oi
        case 4 => MklDnn.MemoryFormat.oihw
      }
      gradWeightPrim.initUser(diffWeight, MklDnn.DataType.f32, weightFormat, engine)
      gradWeightPrim.initInternal(opPrimDesc, MklDnn.Query.diff_weights_pd)
      gradBiasPrim.initUser(diffBias, MklDnn.DataType.f32, MklDnn.MemoryFormat.x, engine)

      val srcs = Array(inputPrim.workPrim(), gradOutputPrim.workPrim())
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(gradWeightPrim.workPrim(), gradBiasPrim.workPrim())

//      if (inputPrim.reorder != 0) {
//        backwardWeightPrimBuffer += inputPrim.reorder
//      }

//      if (gradOutputPrim.reorder != 0) {
//        backwardWeightPrimBuffer += gradOutputPrim.reorder
//      }

      backWeightPrim = MklDnn.PrimitiveCreate2(opPrimDesc,
        srcs, indexes, srcs.length, dsts, dsts.length)

      backwardWeightPrimBuffer += backWeightPrim

      if (gradWeightPrim.reorder != 0) {
        backwardWeightPrimBuffer += gradWeightPrim.reorder
      }
    }

    inputPrim.setHandle(input)
    gradOutputPrim.setHandle(gradOutput)
    gradWeightPrim.setHandle(diffWeight)
    gradBiasPrim.setHandle(diffBias)

    MklDnn.StreamSubmit(stream, backwardWeightPrimBuffer.length, backwardWeightPrimBuffer.toArray)

    inputPrim.releaseHandle()
    gradOutputPrim.releaseHandle()
    gradWeightPrim.releaseHandle()
    gradBiasPrim.releaseHandle()

    gradWeight.add(ev.fromType(1), diffWeight)
    if (withBias) {
      gradBias.add(ev.fromType(1), diffBias)
    }
    val end2 = System.nanoTime()
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.add(ev.negative(learningRate), gradWeight)
    if (withBias) bias.add(ev.negative(learningRate), gradBias)
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.resize(outputSize, inputSize)
    gradWeight.zero()
    if (withBias) {
      gradBias.resize(outputSize)
      gradBias.zero()
    }
  }

  override def clearState() : this.type = {
    super.clearState()
    addBuffer.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }

  override def getParametersTable(): Table = {
    if (null == bias) {
      T(getName() -> T("weight" -> weight, "gradWeight" -> gradWeight))
    } else {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias))
    }
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) {
      return true
    }

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($inputSize -> $outputSize)"
  }
}

object Linear {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]) : Linear[T] = {
    new Linear[T](inputSize, outputSize,
      withBias, wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
