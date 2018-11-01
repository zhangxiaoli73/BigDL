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
package com.intel.analytics.bigdl.utils.tf.loaders

import java.io.{FileReader => JFileReader}
import java.nio.ByteOrder
import java.util.{HashMap => JHashMap}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{SpatialAveragePooling, SpatialCrossMapLRN, Squeeze}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.mkldnn.OperateType.TFElement
import com.intel.analytics.bigdl.utils.mkldnn.{IROperate, OperateType, TFElement}
import com.intel.analytics.bigdl.utils.tf.Context
import com.intel.analytics.bigdl.utils.tf.loaders.Utils._
import org.tensorflow.framework.NodeDef

import scala.collection.JavaConverters._
import scala.reflect.ClassTag


abstract class TensorflowParser() {
  def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder, context: Context[T])
                        (implicit ev: TensorNumeric[T]): TFElement
}

class MaxPool extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    val attributes = node.getAttrMap
    IROperate.SpatialMaxPooling.data_format = getString(attributes, "data_format")
    IROperate.SpatialMaxPooling.strides = getIntList(attributes, "strides")
    IROperate.SpatialMaxPooling.ksize = getIntList(attributes, "ksize")
    new TFElement(node.getName, IROperate.SpatialMaxPooling, node)
  }
}

class AvePool extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    val attributes = node.getAttrMap
    IROperate.SpatialAvePooling.data_format = getString(attributes, "data_format")
    IROperate.SpatialAvePooling.strides = getIntList(attributes, "strides")
    IROperate.SpatialAvePooling.ksize = getIntList(attributes, "ksize")
    IROperate.SpatialAvePooling.countIncludePad = false

    new TFElement(node.getName, IROperate.SpatialAvePooling, node)
  }
}

//class Conv2D extends TensorflowParser {
//  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
//                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
//  ....
//  }
//}

class LRN extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    val size = getInt(node.getAttrMap, "depth_radius")
    val k = getFloat(node.getAttrMap, "bias")
    val alpha = getFloat(node.getAttrMap, "alpha")
    val beta = getFloat(node.getAttrMap, "beta")

    new TFElement(node.getName, OperateType.LRN,
      Map("data_format" -> "NHWC", "size" -> (size * 2 + 1),
        "ksize" -> k, "alpha" -> alpha * (size * 2 + 1), "beta" -> beta),
      node)
  }
}

class Identity extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    val size = getInt(node.getAttrMap, "depth_radius")
    val k = getFloat(node.getAttrMap, "bias")
    val alpha = getFloat(node.getAttrMap, "alpha")
    val beta = getFloat(node.getAttrMap, "beta")

    new TFElement(node.getName, OperateType.Identity, Map(), node)
  }
}

class Squeeze extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {

    var dims = node.getAttrOrThrow("squeeze_dims").getList().getIList()
      .asScala.map(_.toInt + 1).toArray

    dims = if (dims.isEmpty) null else dims

    new TFElement(node.getName, OperateType.Squeeze,
      Map("dims" -> dims, "batchMode" -> false), node)
  }
}
