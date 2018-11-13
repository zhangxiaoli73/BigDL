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

case class IRSpatialMaxPooling(data_format: String, strides: Seq[Int],
                               ksize: Seq[Int], paddingType : String) extends IROperate

case class IRSpatialAvePooling(data_format: String, strides: Seq[Int],
                               ksize: Seq[Int], paddingType : String,
                               countIncludePad: Boolean) extends IROperate

case class IRSpatialConv[T: ClassTag](data_format : String, nInputPlane: Int,
            nOutputPlane: Int, ksize : Seq[Int], strides : Seq[Int], paddingType : String,
            nGroup: Int, weights: Tensor[T] = null, bias: Tensor[T] = null,
            gradWeights: Tensor[T] = null, gradBias: Tensor[T] = null,
            propagateBack: Boolean = true, withBias: Boolean = true) extends IROperate

case class IRSpatialBatchNorm[T: ClassTag](nOutput: Int, weights: Tensor[T],
                                           bias: Tensor[T], initGradWeight: Tensor[T],
                                           initGradBias: Tensor[T], data_format: String) extends IROperate

case class IRIdentity() extends IROperate

case class IRDropout(initP: Double = 0.5, inplace: Boolean = false) extends IROperate

case class IRReLu() extends IROperate

case class IRLinear[T: ClassTag](inputSize: Int,
                          outputSize: Int,
                          withBias: Boolean = true,
                          wRegularizer: Regularizer[T] = null,
                          bRegularizer: Regularizer[T] = null,
                          initWeight: Tensor[T] = null,
                          initBias: Tensor[T] = null,
                          initGradWeight: Tensor[T] = null,
                          initGradBias: Tensor[T] = null) extends IROperate

case class IRSqueeze(dims: Array[Int], batchMode: Boolean) extends IROperate

case class IRLRN(size: Int = 5,
                alpha: Double = 1.0,
                beta: Double = 0.75,
                k: Double = 1.0,
                data_format: String = "NCHW") extends IROperate

case class IRInput(var data_format: String, size: Array[Int]) extends IROperate

case class IROutput() extends IROperate

case class IRSelectTable(dimension: Int) extends IROperate


private[bigdl] class IRElement(
  private var name: String,
  private var op_type: IROperate,
  private var formats: String = "",
  var inputShape: Array[Int] = null,
  var outputShape: Array[Int] = null) {

  private var inputs : Seq[String] = _

  // Get the element name
  final def getName() : String = {
    this.name
  }

  final def getOp() : IROperate = {
    this.op_type
  }
  // option methos
  final def setinput(in: Seq[String]) : Unit = {
    inputs = in
  }

  final def getinput() : Seq[String] = {
    inputs
  }

  final def getformats() : String = {
    formats
  }

  final def setFormats(format: String) : Unit = {
    formats = format
  }
}

private[bigdl] class TFElement(
  private val name: String,
  private val op_type: IROperate,
  private val tf_layer: NodeDef,
  private var formats: String = "",
  var inputShape1: Array[Int] = null,
  var outputShape1: Array[Int] = null)
  extends IRElement(name, op_type, formats, inputShape1, outputShape1) {

  private var layer : NodeDef = tf_layer

  def getLayer() : NodeDef = {
    layer
  }
}

object IRElement {
  def apply(name: String, op_type: IROperate, formats: String = "",
            inputShape: Array[Int] = null, outputShape: Array[Int] = null): IRElement =
    new IRElement(name, op_type, formats, inputShape, outputShape)

//  def apply(name: String, op_type: IROperate, tf_layer: NodeDef, formats: String = "",
//            inputShape: Array[Int] = null, outputShape: Array[Int] = null)
//  : IRElement =
//    new TFElement(name, op_type, tf_layer, formats, inputShape, outputShape)
}