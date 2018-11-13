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
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

sealed class IROperate {
  def name: String = this.getClass.getSimpleName
}

case class IRSpatialMaxPooling(
            kW: Int, kH: Int,
            dW: Int = 1, dH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            format: DataFormat = DataFormat.NCHW) extends IROperate

case class IRSpatialAveragePooling(
            kW: Int, kH: Int,
            dW: Int = 1, dH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            globalPooling: Boolean = false,
            ceilMode: Boolean = false, countIncludePad: Boolean = true,
            divide: Boolean = true, format: DataFormat = DataFormat.NCHW) extends IROperate

case class IRSpatialConvolution[T: ClassTag](
            nInputPlane: Int, nOutputPlane: Int,
            kernelW: Int, kernelH: Int,
            strideW: Int = 1, strideH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            nGroup: Int = 1, propagateBack: Boolean = true,
            wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            withBias: Boolean = true, format: DataFormat = DataFormat.NCHW) extends IROperate


case class IRSpatialBatchNormalization[T: ClassTag](
            nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1,
            affine: Boolean = true,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            dataFormat: DataFormat = DataFormat.NCHW) extends IROperate

case class IRIdentity() extends IROperate

case class IRDropout(initP: Double = 0.5, inplace: Boolean = false) extends IROperate

case class IRReLu() extends IROperate

case class IRLinear[T: ClassTag](
            inputSize: Int,
            outputSize: Int,
            withBias: Boolean = true,
            wRegularizer: Regularizer[T] = null,
            bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null,
            initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null,
            initGradBias: Tensor[T] = null) extends IROperate

case class IRSqueeze(dims: Array[Int], batchMode: Boolean) extends IROperate

case class IRSpatialCrossMapLRN(
            size: Int = 5,
            alpha: Double = 1.0,
            beta: Double = 0.75,
            k: Double = 1.0,
            data_format: DataFormat = DataFormat.NCHW) extends IROperate

case class IRInput(var data_format: String, size: Array[Int]) extends IROperate

case class IROutput() extends IROperate

case class IRSelectTable(dimension: Int) extends IROperate


private[bigdl] class IRElement(
  val name: String,
  val op_type: IROperate,
  private var formats: String = "",
  private var src_layer: Any = null,
  var inputShape: Array[Int] = null,
  var outputShape: Array[Int] = null) {

  private var inputs : Seq[String] = _

  // Get the element name
  def getName() : String = this.name

  def getOp() : IROperate = this.op_type

  final def getFormats() : String = formats

  final def setFormats(format: String) : Unit = {
    formats = format
  }

  final def srcLayer() : Any = src_layer
}

object IRElement {
  def apply(name: String, op_type: IROperate,
            formats: String = "", src_layer: Any = null,
            inputShape: Array[Int] = null, outputShape: Array[Int] = null): IRElement =
    new IRElement(name, op_type, formats, inputShape, outputShape)
}