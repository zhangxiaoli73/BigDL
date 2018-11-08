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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.mkldnn._
import com.intel.analytics.bigdl.utils.tf.Context
import com.intel.analytics.bigdl.utils.tf.loaders.Utils._
import org.tensorflow.framework.NodeDef

import scala.collection.JavaConverters._
import scala.reflect.ClassTag


abstract class TensorflowParser() {
  def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder, context: Context[T])
                        (implicit ev: TensorNumeric[T]): TFElement
}

class tfMaxPool extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    val attributes = node.getAttrMap
    val data_format = getString(attributes, "data_format")
    val strides = getIntList(attributes, "strides")
    val ksize = getIntList(attributes, "ksize")
    val padding = getString(attributes, "padding")
    new TFElement(node.getName, IRSpatialMaxPooling(data_format, strides, ksize, padding), node)
  }
}

class tfAvePool extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    val attributes = node.getAttrMap
    val data_format = getString(attributes, "data_format")
    val strides = getIntList(attributes, "strides")
    val ksize = getIntList(attributes, "ksize")
    val padding = getString(attributes, "padding")
    new TFElement(node.getName,
      IRSpatialAvePooling(data_format, strides, ksize, padding, false), node)
  }
}

class tfLRN extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    val attributes = node.getAttrMap
    val size = getInt(attributes, "depth_radius")
    val k = getFloat(attributes, "bias")
    val alpha = getFloat(attributes, "alpha")
    val beta = getFloat(attributes, "beta")
    val data_format = getString(attributes, "data_format")

    new TFElement(node.getName, IRLRN(size*2 + 1, k, alpha * (size * 2) + 1, beta, data_format),
      node)
  }
}

class tfIdentity extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    new TFElement(node.getName, IRIdentity(), node)
  }
}

class tfSqueeze extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {

    var dims = node.getAttrOrThrow("squeeze_dims").getList().getIList()
      .asScala.map(_.toInt + 1).toArray

    dims = if (dims.isEmpty) null else dims

    new TFElement(node.getName, IRSqueeze(dims, false), node)
  }
}

class tfRelu extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
    new TFElement(node.getName, IRReLu(), node)
  }
}

class tfPlaceholder extends TensorflowParser {
  override def build[T: ClassTag](node: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): TFElement = {
//    val attributes = node.getAttrMap
//    val data_format = getString(attributes, "data_format")

    // todo: ????
    new TFElement(node.getName, IRIdentity(), node)
  }
}