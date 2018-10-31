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

import java.io.{DataInputStream, InputStream, FileReader => JFileReader}
import java.nio.ByteOrder
import java.util
import java.util.{List, HashMap => JHashMap}

import breeze.linalg.reverse
import com.google.protobuf.{CodedInputStream, TextFormat}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{Graph, SpatialAveragePooling, SpatialMaxPooling}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.nn.tf.{AssignGrad, SwitchControlNode, SwitchOps}
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDLUtils}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.tf.{BigDLSessionImpl, Context, Session, TensorflowLoader}
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL._
import com.intel.analytics.bigdl.utils.tf.loaders.TensorflowOpsLoader
import com.intel.analytics.bigdl.utils.tf.loaders.Utils._
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object TensorflowParser {

  // here just support some ops that can be converted to dnn
  def default[T: ClassTag](node: NodeDef, byteOrder: ByteOrder, context: Context[T])
    : TFElement = {
    val name = node.getName
    val op = node.getOp
    val attrs = node.getAttrMap.asScala.toMap

    new TFElement(name, op, attrs.asInstanceOf[Map[String, Any]], node)
  }

  def maxPool[T: ClassTag](node: NodeDef, byteOrder: ByteOrder, context: Context[T])
    : TFElement = {
    val attributes = node.getAttrMap
    val format = getString(attributes, "data_format")
    val strideList = getIntList(attributes, "strides")
    val kernelList = getIntList(attributes, "ksize")

    new TFElement(" ", "MaxPool",
      Map("data_format" -> format, "strides" -> strideList, "ksize" -> kernelList),
      node)
   }
}
