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
import com.intel.analytics.bigdl.nn.{Graph, SpatialAveragePooling}
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

  // here just support some lops that can be converted to dnn
  private def default[T: ClassTag](node: NodeDef, byteOrder: ByteOrder, context: Context[T])
  : TFElement = {
    val name = node.getName
    val op = node.getOp
    val attrs = node.getAttrMap.asScala.toMap

    new TFElement(name, op, attrs.asInstanceOf[Map[String, Any]])
  }

//  private def averagepooing[T: ClassTag](
//    node: NodeDef,
//    byteOrder: ByteOrder,
//    context: Context[T]): TFElement = {
//    val attributes = node.getAttrMap
//    val format = getString(attributes, "data_format")
//    val strideList = getIntList(attributes, "strides")
//    val kernelList = getIntList(attributes, "ksize")
//
//    val (strideH, strideW, ksizeH, ksizeW) = format match {
//      case "NHWC" =>
//        require(strideList(3) == 1, s"not support strides on depth")
//        (strideList(1), strideList(2), kernelList(1), kernelList(2))
//      case "NCHW" =>
//        require(strideList(1) == 1, s"not support strides on depth")
//        (strideList(2), strideList(3), kernelList(2), kernelList(3))
//      case _ =>
//        throw new IllegalArgumentException(s"not supported data format: $format")
//    }
//
//    val (pW, pH) =
//      if (getString(attributes, "padding") == "SAME") {
//        (-1, -1)
//      } else {
//        (0, 0)
//      }
//
//    SpatialAveragePooling[T](ksizeW, ksizeH, strideW, strideH, pW, pH,
//      countIncludePad = false, format = DataFormat(format))
//
//    new TFElement("SpatialAveragePooling", "SpatialAveragePooling",
//      Map("kW" -> ksizeW,
//        "kH" -> ksizeH,
//        "dW" -> strideW,
//      "dH" -> strideH,
//    "padW"-> 0,
//    padH: Int = 0,
//    globalPooling: Boolean = false,
//    ceilMode: Boolean = false,
//    countIncludePad: Boolean = true,
//    divide: Boolean = true,
//    format: DataFormat = DataFormat.NCHW))
//  }
}
