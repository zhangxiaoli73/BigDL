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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.reflect.ClassTag
import RandomGenerator._
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer

/**
 * The `Linear` module applies a linear transformation to the input data,
 * i.e. `y = Wx + b`. The `input` given in `forward(input)` must be either
 * a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
 * have the size of `inputSize`. If it is a matrix, then each row is assumed to be
 * an input sample of given batch (the number of rows means the batch size and
 * the number of columns should be equal to the `inputSize`).
 *
 * @param inputSize the size the each input sample
 * @param outputSize the size of the module output of each sample
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */
@SerialVersionUID( 359656776803598943L)
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


  var needPack : Boolean = false
  var packMem: Long = 0
  var packAgain = true
  def pack(): Unit = {
    needPack = true
  }
  def resetPack(): Unit = {
    packAgain = true
  }

  def beginPack(): Unit = {
//    packAgain = true
//    packMem = 0L
  }

  def setPackValue(value : Long): Unit = {
    packMem = value
  }

  def packForward(input: Tensor[T], output: Tensor[T]): Unit = {
    if (MKL.isMKLLoaded && weight.isContiguous() && (ev.getType() == FloatType)) {
      val b : Array[Float] = input.asInstanceOf[Tensor[Float]].storage().array()
      val a : Array[Float] = weight.asInstanceOf[Tensor[Float]].storage().array()
      val c: Array[Float] = output.asInstanceOf[Tensor[Float]].storage().array()

      val m = weight.size(1)
      val k = weight.size(2)
      val n = input.size(1)
      val alpha = 1
      val beta = 0
      val lda: Int = weight.size(2)
      val ldb: Int = input.size(2)
      val ldc: Int = output.size(2)

//      println(s"in linear $packMem")

      if (packMem == 0) {
        println("memory only once")
        packMem = MKL.sgemmAlloc('A', m, n, k)
      }

      if (packAgain) {
        println("pack again only once")
        MKL.sgemmPack('A', 'T', m, n, k, alpha, a, 0, lda, packMem)
        packAgain = false
      }
      MKL.sgemmCompute('P', 'N', m, n, k, a, 0, lda, b, 0, ldb, beta, c, 0, ldc, packMem)

      // MKL.vsgemm('T', 'N', m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc)
    }
  }

  def packBackward(input: Tensor[T], output: Tensor[T]): Unit = {
    if (MKL.isMKLLoaded && weight.isContiguous() && (ev.getType() == FloatType)) {
      val b : Array[Float] = input.asInstanceOf[Tensor[Float]].storage().array()
      val a : Array[Float] = weight.asInstanceOf[Tensor[Float]].storage().array()
      val c: Array[Float] = output.asInstanceOf[Tensor[Float]].storage().array()

      val m = weight.size(2)
      val k = weight.size(1)
      val n = input.size(1)
      val alpha = 1
      val beta = 0
      val lda: Int = weight.size(2)
      val ldb: Int = input.size(2)
      val ldc: Int = output.size(2)

      if (packMem == 0) {
        println("memory backward only once")
        packMem = MKL.sgemmAlloc('A', m, n, k)
      }
      if (packAgain) {
        println("pack again only once backward")
        MKL.sgemmPack('A', 'N', m, n, k, alpha, a, 0, lda, packMem)
        packAgain = false
      }
      MKL.sgemmCompute('P', 'N', m, n, k, a, 0, lda, b, 0, ldb, beta, c, 0, ldc, packMem)

      // MKL.vsgemm('N', 'N', m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc)
    }
  }

  def freepack(): Unit = {
    MKL.sgemmFree(packMem)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
      s"input dim ${input.dim()}")

    // weight.fill(ev.fromType(0.01))

    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    }
    else if (input.dim() == 2) {
      val nFrame = input.size(1)
      val nElement = output.nElement
      val t = Array(nFrame, weight.size(1))
      output.resize(t)
      if (output.nElement() != nElement) {
        output.zero()
      }

      if (addBuffer.nElement() != nFrame) {
        addBuffer.resize(Array(nFrame)).fill(ev.one)
      }

      val s1 = System.nanoTime()
      if (needPack) {
        packForward(input, output)
      } else {
        output.addmm(ev.zero, output, ev.one, input, weight.t)
      }
      val end1 = System.nanoTime() - s1
//      println(s"$needPack forward time ${end1/1e9}")
      if (withBias) output.addr(ev.one, addBuffer, bias)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
    s"input dim ${input.dim()}")

    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    if (input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      if (needPack) {
        packBackward(gradOutput, gradInput)
      } else {
        gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
      }
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch +
    s"input dim ${input.dim()}")

    gradWeight.resize(outputSize, inputSize)
    if (withBias) {
      gradBias.resize(outputSize)
    }

    if (input.dim() == 1) {
      if (scaleW != 0) {
        gradWeight.addr(ev.fromType[Double](scaleW), gradOutput, input)
      }

      if (withBias && scaleB != 0) {
        gradBias.add(ev.fromType[Double](scaleB), gradOutput)
      }
    }
    else if (input.dim() == 2) {
      if (scaleW != 0) {
        gradWeight.addmm(ev.fromType[Double](scaleW), gradOutput.t, input)
      }

      if (withBias && scaleB != 0) {
        gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t, addBuffer)
      }
    }

    if (null != wRegularizer && scaleW != 0) {
      println("update weight")
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
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
