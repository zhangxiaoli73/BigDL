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

import java.util

import com.intel.analytics.bigdl.nn.{DynamicGraph, Graph, StaticGraph, mkldnn}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.nn.tf.MaxPool
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, DataType}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import com.intel.analytics.bigdl.utils.caffe.CaffeConversionException
import com.intel.analytics.bigdl.utils.serializer.DeserializeContext
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer.getCostructorMirror
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter.{ArrayConverter, CustomConverterDelegator, NameListConverter}
import com.intel.analytics.bigdl.utils.serializer.converters._
import com.intel.analytics.bigdl.utils.tf.loaders.TensorflowOpsLoader

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.{ClassTag, ManifestFactory}
import scala.reflect.runtime.universe


class IRLayer2Blas[T: ClassTag](implicit ev: TensorNumeric[T]) {

  private val IR2BlasMap = new mutable.HashMap[String, (IRElement) => Module[T]]

  mapInit()

  def enableConvert(layer: IRElement) : Boolean = {
    val layerType = layer.getOp()
    if (IR2BlasMap.contains(layerType.name)) true
    else false
  }

  def getFiledNameAndValues(o: Object): mutable.HashMap[String, AnyRef] = {
    val c = o.getClass
    val fields = c.getDeclaredFields
    val values = new mutable.HashMap[String, AnyRef]()
    var i = 0
    while (i < fields.length) {
      val field = fields(i)
      val name = field.getName
      field.setAccessible(true)
      values(name) = field.get(o)
      i += 1
    }
    values
  }

//  def getAttributeValue[T : ClassTag](attribute: AttrValue)
//                                              (implicit ev: TensorNumeric[T]) : AnyRef = {
//    attribute.getDataType match {
//      case DataType.INT32 => Integer.valueOf(attribute.getInt32Value)
//      case DataType.INT64 => Long.box(attribute.getInt64Value)
//      case DataType.DOUBLE => Double.box(attribute.getDoubleValue)
//      case DataType.FLOAT => Float.box(attribute.getFloatValue)
//      case DataType.STRING => attribute.getStringValue
//      case DataType.BOOL => Boolean.box(attribute.getBoolValue)
//      case DataType.REGULARIZER => RegularizerConverter.getAttributeValue(context, attribute)
//      case DataType.TENSOR => TensorConverter.getAttributeValue(context, attribute)
//      case DataType.VARIABLE_FORMAT =>
//        VariableFormatConverter.getAttributeValue(context, attribute)
//      case DataType.INITMETHOD => InitMethodConverter.getAttributeValue(context, attribute)
//      case DataType.MODULE => ModuleConverter.getAttributeValue(context, attribute)
//      case DataType.NAME_ATTR_LIST => NameListConverter.getAttributeValue(context, attribute)
//      case DataType.ARRAY_VALUE => ArrayConverter.getAttributeValue(context, attribute)
//      case DataType.DATA_FORMAT => DataFormatConverter.getAttributeValue(context, attribute)
//      case DataType.CUSTOM => CustomConverterDelegator.getAttributeValue(context, attribute)
//      case DataType.SHAPE => ShapeConverter.getAttributeValue(context, attribute)
//      case _ => throw new IllegalArgumentException
//        (s"${attribute.getDataType} can not be recognized")
//    }
//  }


  def convertIRLayer(layer : IRElement) : Module[T] = {
    val name = layer.getOp().name
    // todo :???
    val cls = Class.forName("com.intel.analytics.bigdl.nn." + name.substring(3))
    val nameAndValues = getFiledNameAndValues(layer.getOp())
    val constructorMirror = getCostructorMirror(cls)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams.map(_.size).sum)

    var i = 0
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (ptype <:< universe.typeOf[ClassTag[_]]||
          ptype.typeSymbol == universe.typeOf[ClassTag[_]].typeSymbol) {
          args(i) = ManifestFactory.Float
        } else if (ptype <:< universe.typeOf[TensorNumeric[_]]
          || ptype.typeSymbol == universe.typeOf[TensorNumeric[_]].typeSymbol) {
          args(i) = TensorNumeric.NumericFloat
        } else {
          val value = nameAndValues.get(name).getOrElse(null)
          args(i) = value
        }
        i += 1
      })
    })
    val blasLayer = constructorMirror.apply(args : _*).
      asInstanceOf[Module[T]]
    blasLayer
  }

  private def mapInit(): Unit = {
    IR2BlasMap("IRSpatialConv") = fromConv
    IR2BlasMap("IRSpatialMaxPooling") = fromMaxPooling
    // todo: not complete
  }

  private def fromConv(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromMaxPooling(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromSelectTable(node: IRElement) : Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromSpatialBatchNormalization(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")
}

object IRLayer2Blas {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRLayer2Blas[T] = new IRLayer2Blas
}