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

import com.intel.analytics.bigdl.nn.ReLU
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, ModuleSerializer, SerializeContext}

import scala.reflect.ClassTag
import scala.reflect.runtime._

sealed class IROperate[T: ClassTag] extends Serializable {
  val tag: ClassTag[T] = scala.reflect.classTag[T]
  val numerics: TensorNumeric[T] = tag match {
    case ClassTag.Float => TensorNumeric.NumericFloat.asInstanceOf[TensorNumeric[T]]
    case ClassTag.Double => TensorNumeric.NumericDouble.asInstanceOf[TensorNumeric[T]]
    case _ => throw new IllegalArgumentException(s"not supported class tag: ${tag}")
  }
  def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array(scala.reflect.classTag[T]), Array(numerics))
  }
  def name: String = this.getClass.getSimpleName
}

case class IRSpatialMaxPooling[T: ClassTag](
            kW: Int, kH: Int,
            dW: Int = 1, dH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            format: DataFormat = DataFormat.NCHW, ceilMode: Boolean = false) extends IROperate[T]

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

case class IRSpatialShareConvolution[T: ClassTag](
            nInputPlane: Int, nOutputPlane: Int,
            kernelW: Int, kernelH: Int,
            strideW: Int = 1, strideH: Int = 1,
            padW: Int = 0, padH: Int = 0,
            nGroup: Int = 1, propagateBack: Boolean = true,
            wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            withBias: Boolean = true) extends IROperate[T]

case class IRSpatialBatchNormalization[T: ClassTag](
            nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1,
            affine: Boolean = true,
            initWeight: Tensor[T] = null, initBias: Tensor[T] = null,
            initGradWeight: Tensor[T] = null, initGradBias: Tensor[T] = null,
            dataFormat: DataFormat = DataFormat.NCHW,
            runningMean: Tensor[T] = null, runningVar: Tensor[T] = null) extends IROperate[T]

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
            format: DataFormat = DataFormat.NCHW) extends IROperate[T]

//case class IRSoftMax[T: ClassTag]() extends IROperate[T]

case class IRSelectTable[T: ClassTag](dimension: Int) extends IROperate[T]

case class IRCAddTable[T: ClassTag, D: ClassTag](inplace: Boolean = false) extends IROperate[T]

case class IRJoinTable[T: ClassTag](dimension: Int,
                                    nInputDims: Int = 0) extends IROperate[T]

case class IRConcatTable[T: ClassTag]() extends IROperate[T]

case class IRInput[T: ClassTag]() extends IROperate[T]

case class IRBlasModule[T: ClassTag](
             model: AbstractModule[Activity, Activity, T]) extends IROperate[T]

private[bigdl] class IRElement[T: ClassTag](
  val name: String,
  val op_type: IROperate[T],
  var formats: String = "",
  private var weights: Tensor[T] = null,
  private var gradWeights: Tensor[T] = null) extends Serializable {

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

//object IRElement extends ModuleSerializable {
//  def apply[T: ClassTag](name: String, op_type: IROperate[T],
//            formats: String = "", weights: Tensor[T] = null,
//            bias: Tensor[T] = null): IRElement[T] =
//    new IRElement[T](name, op_type, formats, weights, bias)
//
//  override def doLoadModule[T: ClassTag](context: DeserializeContext)
//      (implicit ev: TensorNumeric[T]) : IRElement[T] = {
//
//    val attrMap = context.bigdlModule.getAttrMap
//    val name = DataConverter.getAttributeValue(context, attrMap.get("name")).
//      asInstanceOf[String]
//    val op_type = DataConverter.getAttributeValue(context, attrMap.get("op_type")).
//      asInstanceOf[IROperate[T]]
//    val formats = DataConverter.getAttributeValue(context, attrMap.get("formats")).
//      asInstanceOf[String]
//    val weights = DataConverter.getAttributeValue(context, attrMap.get("weights")).
//      asInstanceOf[Tensor[T]]
//    val gradWeights = DataConverter.getAttributeValue(context, attrMap.get("bias")).
//      asInstanceOf[Tensor[T]]
//
//    IRElement(name, op_type, formats, weights, gradWeights)
//  }
//
//  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
//                                              reshapeBuilder : BigDLModule.Builder)
//                                             (implicit ev: TensorNumeric[T]) : Unit = {
//
//    val reshape = context.moduleData.module.asInstanceOf[IRElement[T]]
//
//    val nameBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, nameBuilder, reshape.name,
//      universe.typeOf[String])
//    reshapeBuilder.putAttr("name", nameBuilder.build)
//
////    val opBuilder = AttrValue.newBuilder
////    DataConverter.setAttributeValue(context, opBuilder, reshape.getOp(),
////      IROperate[T].typeSignature)
////    reshapeBuilder.putAttr("op_type", opBuilder.build)
//
//    val formatsBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, formatsBuilder, reshape.formats,
//      universe.typeOf[String])
//    reshapeBuilder.putAttr("formats", formatsBuilder.build)
//
//    val weightsBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, weightsBuilder, reshape.weights,
//      ModuleSerializer.tensorType)
//    reshapeBuilder.putAttr("weights", weightsBuilder.build)
//
//    val biasBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, biasBuilder, reshape.gradWeights,
//      ModuleSerializer.tensorType)
//    reshapeBuilder.putAttr("bias", biasBuilder.build)
//  }
//}
