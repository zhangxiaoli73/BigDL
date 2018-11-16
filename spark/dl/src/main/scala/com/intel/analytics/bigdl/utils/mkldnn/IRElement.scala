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

package com.intel.analytics.bigdl.utils.mkldnn

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

sealed class IROperate[T] {
  def name: String = this.getClass.getSimpleName
}

case class IRSpatialMaxPooling[T: ClassTag](
            kW: Int, kH: Int,
            dW: Int = 1, dH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            format: DataFormat = DataFormat.NCHW) extends IROperate[T]

case class IRSpatialAveragePooling[T: ClassTag](
            kW: Int, kH: Int,
            dW: Int = 1, dH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            globalPooling: Boolean = false,
            ceilMode: Boolean = false, countIncludePad: Boolean = true,
            divide: Boolean = true, format: DataFormat = DataFormat.NCHW) extends IROperate[T]

case class IRSpatialConvolution[T: ClassTag](
            nInputPlane: Int, nOutputPlane: Int,
            kernelW: Int, kernelH: Int,
            strideW: Int = 1, strideH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            nGroup: Int = 1, propagateBack: Boolean = true,
            wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            withBias: Boolean = true, format: DataFormat = DataFormat.NCHW) extends IROperate[T]


case class IRSpatialBatchNormalization[T: ClassTag](
            nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1,
            affine: Boolean = true,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            dataFormat: DataFormat = DataFormat.NCHW) extends IROperate[T]

case class IRIdentity[T: ClassTag]() extends IROperate[T]

case class IRDropout[T: ClassTag](initP: Double = 0.5, inplace: Boolean = false,
                                  scale: Boolean = true) extends IROperate[T]

case class IRReLU[T: ClassTag](ip: Boolean = false) extends IROperate[T]

case class IRLinear[T: ClassTag](
            inputSize: Int,
            outputSize: Int,
            withBias: Boolean = true,
            wRegularizer: Regularizer[T] = null,
            bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null,
            initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null,
            initGradBias: Tensor[T] = null) extends IROperate[T]

case class IRSqueeze[T: ClassTag](dims: Array[Int], batchMode: Boolean) extends IROperate[T]

case class IRSpatialCrossMapLRN[T: ClassTag](
            size: Int = 5,
            alpha: Double = 1.0,
            beta: Double = 0.75,
            k: Double = 1.0,
            data_format: DataFormat = DataFormat.NCHW) extends IROperate[T]

case class IRInput[T: ClassTag](var data_format: String, size: Array[Int]) extends IROperate[T]

case class IRSelectTable[T: ClassTag](dimension: Int) extends IROperate[T]

case class IRReshape[T: ClassTag](
            size: Array[Int], batchMode: Option[Boolean] = None) extends IROperate[T]

case class IRView[T: ClassTag]() extends IROperate[T]

case class IRThreshold[T](th: Double = 1e-6, v: Double = 0.0,
                          ip: Boolean = false) extends IROperate[T]

case class IRLogSoftMax[T: ClassTag]() extends IROperate[T]

private[bigdl] class IRElement[T: ClassTag](
  val name: String,
  val op_type: IROperate[T],
  var formats: String = "",
  private var weights: Tensor[T] = null,
  private var gradWeights: Tensor[T] = null) {

  /**
   * set weight and bias
   */
  def setWeights(weightsAndBias: Tensor[T]) : Unit = {
    weights = weightsAndBias
  }

  /**
   * set gradWeight and gradbias
   */
  def setGradWeights(gradWeightsAndGradBias: Tensor[T]) : Unit = {
    gradWeights = gradWeightsAndGradBias
  }

  def getParameters(): (Tensor[T], Tensor[T]) = (weights, gradWeights)

  def getName() : String = this.name

  def getOp() : IROperate[T] = this.op_type

  final def getFormats() : String = formats

  final def setFormats(format: String) : Unit = {
    formats = format
  }
}

object IRElement {
  def apply[T: ClassTag](name: String, op_type: IROperate[T],
            formats: String = "", weights: Tensor[T] = null,
            bias: Tensor[T] = null): IRElement[T] =
    new IRElement[T](name, op_type, formats, weights, bias)
}