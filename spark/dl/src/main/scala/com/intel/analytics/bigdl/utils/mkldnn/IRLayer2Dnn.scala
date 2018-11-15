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

import java.nio.ByteOrder
import java.util
import java.util.List

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import spire.macros.Auto.scala

class IRLayer2Dnn {

  // converter function mappings
  private val IR2DnnMap = new mutable.HashMap[String, (IRElement[Float]) => Module[Float]]
  mapInit()

  private def mapInit(): Unit = {
    IR2DnnMap("IRSpatialConvolution") = fromConv
    IR2DnnMap("IRSpatialMaxPooling") = fromMaxPooling
    IR2DnnMap("IRSpatialAveragePooling") = fromAvgPooling
    IR2DnnMap("IRSpatialBatchNormalization") = fromSbn
    IR2DnnMap("IRLinear") = fromLinear
  }

  def enableConvert(layer: IRElement[Float]) : Boolean = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name)) return true
    try {
      val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn" + name.substring(2))
      true
    } catch {
      case e: Throwable =>
        false
    }
  }

  def convertIRLayer(layer : IRElement[Float]) : Module[Float] = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name)) {
      IR2DnnMap(name)(layer)
    } else {
      val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn" + name.substring(2))
      val dnnLayer = ReflectionUtils.reflectFromIR(layer, cls)
      dnnLayer
    }
  }

  private def fromConv(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialConvolution[Float]]
    val nInputPlane = t.nInputPlane.asInstanceOf[Int]
    val nOutputPlane = t.nOutputPlane.asInstanceOf[Int]
    val kernelW = t.kernelW
    val kernelH = t.kernelH
    val strideW = t.strideW
    val strideH = t.strideH
    val padW = t.padW
    val padH = t.padH

    if (t.format == DataFormat("NCHW")) {
      val initWeight = t.initWeight
      val initBias = t.initWeight
      val initGradWeight = t.initGradWeight
      val initGradBias = t.initGradBias
      val layer = mkldnn.SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, initBias = initBias, initGradBias = initGradBias,
        initWeight = initWeight, initGradWeight = initGradWeight)
      val params = node.getParameters()
      val params2 = layer.getParameters()
      if (params._1 != null) params2._1.copy(params._1)
      if (params._2 != null) params2._2.copy(params._2)

      layer

    } else {
      // from NHWC -> NCHW
      val initWeight = if (t.initWeight != null) {
        t.initWeight.transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous().clone()
      } else null
      val initBias = t.initBias
      val initGradWeight = if (t.initGradWeight != null) {
        t.initGradWeight.transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous().clone()
      } else null
      val initGradBias = t.initGradWeight
      val layer = mkldnn.SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, initBias = initBias, initGradBias = initGradBias,
        initWeight = initWeight, initGradWeight = initGradWeight)

      // todo: handle NHWC
      val params = node.getParameters()
      val params2 = layer.getParameters()
      if (params._1 != null) params2._1.copy(params._1)
      if (params._2 != null) params2._2.copy(params._2)

      layer
    }
  }

  private def fromMaxPooling(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialMaxPooling[Float]]
    val kernelW = t.kW
    val kernelH = t.kH
    val strideW = t.dW
    val strideH = t.dH
    val padW = t.padW
    val padH = t.padH
    mkldnn.MaxPooling(kernelW, kernelH, strideW, strideH, padW, padH)
  }

  private def fromAvgPooling(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialAveragePooling[Float]]
    val kernelW = t.kW
    val kernelH = t.kH
    val strideW = t.dW
    val strideH = t.dH
    val padW = t.padW
    val padH = t.padH
    mkldnn.AvgPooling(kernelW, kernelH, strideW, strideH, padW, padH)
  }

  // todo :not corret
  private def fromSbn(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialBatchNormalization[Float]]
    val nOutput = t.nOutput
    val eps = t.eps
    val momentum = t.momentum
    val initWeight = t.initWeight
    val initBias = t.initBias
    val initGradWeight = t.initGradWeight
    val initGradBias = t.initGradBias

    val layer = mkldnn.SpatialBatchNormalization(nOutput, eps, momentum,
      true, initWeight, initBias, initGradWeight, initGradBias)

    val params = node.getParameters()
    if (params._1 != null) layer.weightAndBias.copy(params._1)
    if (params._2 != null) layer.gradWeightAndBias.copy(params._2)
    layer
  }

  private def fromLinear(node: IRElement[Float]) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRLinear[Float]]
    val inputSize = t.inputSize
    val outputSize = t.outputSize
    val withBias: Boolean = true
    val initWeight: Tensor[Float] = t.initWeight
    val initBias: Tensor[Float] = t.initBias
    val initGradWeight: Tensor[Float] = t.initGradWeight
    val initGradBias: Tensor[Float] = t.initGradBias


    val layer = mkldnn.Linear(inputSize, outputSize, withBias, initWeight,
      initBias, initGradWeight, initGradBias)

    val params = node.getParameters()
    val params2 = layer.getParameters()
    if (params._1 != null) params2._1.copy(params._1)
    if (params._2 != null) params2._2.copy(params._2)

    layer
  }
}

object IRLayer2Dnn {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRLayer2Dnn = new IRLayer2Dnn
}