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
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.{AlignedStorage, MklDnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class SpatialBatchNormalization[T: ClassTag](
  val nOutput: Int,
  val eps: Double = 1e-5,
  val momentum: Double = 0.1,
  val affine: Boolean = true,
  private val initWeight: Tensor[T] = null,
  private val initBias: Tensor[T] = null,
  private val initGradWeight: Tensor[T] = null,
  private val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {
  val mean: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))
  val variance: MklDnnTensor[T] = MklDnnTensor[T](Array(nOutput))

  val all = createParams(initWeight, initBias)
  val gradAll = createParams(initGradWeight, initGradBias)
  val diffAll = createParams(initGradWeight, initGradBias)

  @transient var engine = 0L
  @transient var stream = 0L

  @transient var forwardPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var backwardPrims: ArrayBuffer[Long] = ArrayBuffer.empty
  @transient var forwardPrimDesc = 0L

  object OpPrim {
    val input, output, weightAndBias, mean, variance,
        diffInput, diffOutput, diffWeightAndBias = new MemoryPrimitive[T]()
  }

  // TODO train and inference mode

  @transient var internalInput, internalOutput: MklDnnTensor[T] = _

  val defaultFormat = MklDnn.MemoryFormat.nchw

  private def toMklDnnTensor(t: Tensor[T]): MklDnnTensor[T] = t.asInstanceOf[MklDnnTensor[T]]

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    input.getTensorType match {
      case MklDnnType => internalInput = input.asInstanceOf[MklDnnTensor[T]]
      case _ =>
        if (internalInput == null) {
          internalInput = MklDnnTensor[T](input.size())
        } else if (internalInput.size().deep != input.size().deep) {
          internalInput.resize(input.size())
        }

        internalInput.copy(input)
    }

    if (!output.isInstanceOf[MklDnnTensor[T]]) {
      output = MklDnnTensor[T](input.size())
    }

    output.resizeAs(input)

    if (forwardPrims.isEmpty) {
      engine = this.getDnnEngine(0)
      stream = this.getStream()

      val srcMemDesc = if (input.getPrimitiveDesc() == 0L) {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          MklDnn.DataType.f32, defaultFormat) // TODO
      } else {
        MklDnnOps.primitiveDescQueryMemory(input.getPrimitiveDesc())
      }

      val opDesc = MklDnn.BatchNormForwardDescInit(MklDnn.PropKind.forward,
        srcMemDesc, eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)
      forwardPrimDesc = opPrimDesc

      val dataFormat = defaultFormat
      val paramsFormat = MklDnn.MemoryFormat.x
      val dataType = MklDnn.DataType.f32
      OpPrim.input.initUser(input, dataType, dataFormat, engine)
      OpPrim.input.initInternal(opPrimDesc, MklDnn.Query.src_pd)
      OpPrim.output.initUser(output, opPrimDesc, MklDnn.Query.dst_pd, 0)

      // because they're 1-d, so we need not to initialize it.
      OpPrim.weightAndBias.initUser(all, dataType, paramsFormat, engine)
      OpPrim.mean.initUser(mean, dataType, paramsFormat, engine)
      OpPrim.variance.initUser(variance, dataType, paramsFormat, engine)

      val srcs = Array(OpPrim.input.workPrim(), OpPrim.weightAndBias.workPrim())
      val indexes = Array.fill(srcs.length)(0)
      val dsts = Array(OpPrim.output.workPrim(), OpPrim.mean.workPrim(), OpPrim.variance.workPrim())

      if (OpPrim.input.reorder != 0L) {
        forwardPrims += OpPrim.input.reorder
      }

      forwardPrims += MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)
    }

    OpPrim.input.setHandle(internalInput)
    OpPrim.weightAndBias.setHandle(all)
    OpPrim.output.setHandle(toMklDnnTensor(output), needUpdate = true)
    OpPrim.mean.setHandle(mean, needUpdate = true)
    OpPrim.variance.setHandle(variance, needUpdate = true)

    MklDnnOps.submit(stream, forwardPrims.toArray)
    require(mean.storage().asInstanceOf[AlignedStorage[T]].needConversion == true)

    output
  }

  @transient var internalGradInput, internalGradOutput: MklDnnTensor[T] = _
  def backward1(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isInstanceOf[MklDnnTensor[T]]) {
      gradInput = MklDnnTensor[T](input.size())
    }
    gradInput.resizeAs(input)

    gradOutput.getTensorType match {
      case MklDnnType => internalGradOutput = gradOutput.asInstanceOf[MklDnnTensor[T]]
      case _ =>
        if (internalGradOutput == null) {
          internalGradOutput = MklDnnTensor[T](input.size())
        } else if (internalGradOutput.size().deep != input.size().deep) {
          internalGradOutput.resize(input.size())
        }

        internalGradOutput.copy(gradOutput)
    }

    if (backwardPrims.isEmpty) {
      val srcMemDesc = if (input.getPrimitiveDesc() == 0) {
        MklDnn.MemoryDescInit(input.dim(), input.size(),
          MklDnn.DataType.f32, defaultFormat)
      } else {
        MklDnnOps.primitiveDescQueryMemory(input.getPrimitiveDesc())
      }

      val diffDstMemDesc = if (gradOutput.getPrimitiveDesc() == 0L) {
        MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
          MklDnn.DataType.f32, defaultFormat)
      } else {
        MklDnnOps.primitiveDescQueryMemory(gradOutput.getPrimitiveDesc())
      }

      val desc = MklDnn.BatchNormBackwardDescInit(MklDnn.PropKind.backward,
        diffDstMemDesc, srcMemDesc, eps.toFloat, MklDnn.BatchNormFlag.mkldnn_use_scaleshift)
      val primDesc = MklDnn.PrimitiveDescCreate(desc, engine, forwardPrimDesc)

      val dataFormat = defaultFormat
      val paramsFormat = MklDnn.MemoryFormat.x
      val dataType = MklDnn.DataType.f32

      OpPrim.diffOutput.initUser(gradOutput, dataType, dataFormat, engine)
      OpPrim.diffOutput.initInternal(primDesc, MklDnn.Query.diff_dst_pd)
      OpPrim.diffWeightAndBias.initUser(diffAll, dataType, paramsFormat, engine)
      OpPrim.diffInput.initUser(gradInput, primDesc, MklDnn.Query.diff_src_pd, 0)

      val dataSrcs = Array(OpPrim.input.workPrim(), OpPrim.mean.workPrim(),
        OpPrim.variance.workPrim(), OpPrim.diffOutput.workPrim(),
        OpPrim.weightAndBias.workPrim())
      val dataIndexes = Array.fill(dataSrcs.length)(0)
      val dataDsts = Array(OpPrim.diffInput.workPrim(), OpPrim.diffWeightAndBias.workPrim())

      if (OpPrim.diffOutput.reorder != 0) {
        backwardPrims += OpPrim.diffOutput.reorder
      }

      backwardPrims += MklDnn.PrimitiveCreate2(primDesc, dataSrcs, dataIndexes, dataSrcs.length,
        dataDsts, dataDsts.length)

      if (OpPrim.diffWeightAndBias.reorder != 0) {
        backwardPrims += OpPrim.diffWeightAndBias.reorder
      }
    }

    OpPrim.input.setHandle(internalInput)
    OpPrim.weightAndBias.setHandle(all)
    OpPrim.mean.setHandle(mean)
    OpPrim.variance.setHandle(variance)
    OpPrim.diffOutput.setHandle(internalGradOutput)
    OpPrim.diffInput.setHandle(toMklDnnTensor(gradInput), needUpdate = true)
    OpPrim.diffWeightAndBias.setHandle(diffAll, needUpdate = true)

    MklDnnOps.submit(stream, backwardPrims.toArray)

    gradAll.add(diffAll)

    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    backward1(input, gradOutput)
  }

  // there's no relavant accGrasdParameters in mkl-dnn. we use @backward instead of
  // @updateGradInput and @accGradParameters
  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
  }

  private type Params[R] = (Tensor[R], Tensor[R], Tensor[R])
  // in mkl dnn, the weight and bias should be all in the same array
  private def createParams(initWeight: Tensor[T], initBias: Tensor[T]): MklDnnTensor[T] = {
    val weightAndBias: MklDnnTensor[T] = if (affine) {
      MklDnnTensor[T](Array(2 * nOutput))
    } else {
      null
    }

    val concat = Tensor[T]().resize(Array(2, nOutput)).fill(ev.fromType(0))

    if (initWeight != null) {
      require(initWeight.size(1) == nOutput)
      concat.select(1, 1).copy(initWeight)
    }

    if (initBias != null) {
      require(initBias.size(1) == nOutput)
      concat.select(1, 2).copy(initBias)
    }

    weightAndBias.copy(concat.view(Array(2 * nOutput)))
    weightAndBias
  }

  override def zeroGradParameters(): Unit = {
    if (affine) {
      gradAll.zero()
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (affine) {
      (Array(all), Array(gradAll))
    } else {
      null
    }
  }

  override def getParametersTable(): Table = {
    if (affine) {
      T(getName() -> T("weight" -> all,
        "gradWeight" -> gradAll,
        "runningMean" -> mean, "runningVar" -> variance))
    } else {
      T(getName() -> T("runningMean" -> mean, "runningVar" -> variance))
    }
  }

  override def toString(): String = {
    s"nn.BatchNormalization($nOutput, $eps, $momentum, $affine)"
  }
}

object SpatialBatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null)
    (implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {

    new SpatialBatchNormalization[T](
      nOutput, eps, momentum, affine, initWeight, initBias, initGradWeight, initGradBias)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
    affine: Option[Int])(implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {
    new SpatialBatchNormalization[T](nOutput = affine.getOrElse(1), affine = affine.isDefined)
  }
}
