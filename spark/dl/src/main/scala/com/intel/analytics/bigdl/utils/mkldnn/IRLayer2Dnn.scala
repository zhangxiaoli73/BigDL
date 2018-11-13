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
  private val IR2DnnMap = new mutable.HashMap[String, (IRElement) => Module[Float]]

  mapInit()

  def enableConvert(layer: IRElement) : Boolean = {
    val layerOp = layer.getOp()
    if (IR2DnnMap.contains(layerOp.name)) true
    else false
  }

  def convertIRLayer(layer : IRElement) : Module[Float] = {
    val layerType = layer.getOp()
    require(IR2DnnMap.contains(layerType.name), s"not support convert ${layerType} to dnn layer")
    IR2DnnMap(layerType.name)(layer)
  }

  private def mapInit(): Unit = {
    IR2DnnMap("IRInput") = fromInput
    IR2DnnMap("IRReLu") = fromRelu
    IR2DnnMap("IRDropout") = fromDropOut
    IR2DnnMap("IRIdentity") = fromIdentity
    IR2DnnMap("IRSpatialConvolution") = fromConv
    IR2DnnMap("IRSpatialMaxPooling") = fromMaxPooling
    IR2DnnMap("IRSpatialAveragePooling") = fromAvgPooling
    IR2DnnMap("IRSpatialBatchNormalization") = fromSbn
    IR2DnnMap("IRSqueeze") = fromSqueeze
    // IR2DnnMap("IRLinear") = fromLinear
  }

  private def fromRelu(node: IRElement) : Module[Float] = mkldnn.ReLU()

  private def fromDropOut(node: IRElement) : Module[Float] = mkldnn.Dropout()

  private def fromIdentity(node: IRElement) : Module[Float] = mkldnn.Identity[Float]()

  private def fromSqueeze(node: IRElement) : Module[Float] = {
    // mkldnn.Identity[Float]()
    val t = node.getOp().asInstanceOf[IRSqueeze]
    val s = new Squeeze[Float](t.dims, t.batchMode)
    BigDL2DnnWrapper(s.asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]], "")
  }

  private def fromInput(node: IRElement) : Module[Float] = {
    // todo: not right
    mkldnn.Identity[Float]()
  }

  private def fromConv(node: IRElement) : Module[Float] = {
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
      mkldnn.SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, initBias = initBias, initGradBias = initGradBias,
        initWeight = initWeight, initGradWeight = initGradWeight)
    } else {
      // from NHWC -> NCHW
      val initWeight = t.initWeight.
        transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous().clone()
      val initBias = t.initBias
      val initGradWeight = t.initGradWeight.
        transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous().clone()
      val initGradBias = t.initGradWeight
      mkldnn.SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, initBias = initBias, initGradBias = initGradBias,
        initWeight = initWeight, initGradWeight = initGradWeight)
    }
  }

  private def fromMaxPooling(node: IRElement) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialMaxPooling]
    val kernelW = t.kW
    val kernelH = t.kH
    val strideW = t.dW
    val strideH = t.dH
    val padW = t.padW
    val padH = t.padH
    mkldnn.MaxPooling(kernelW, kernelH, strideW, strideH, padW, padH)
  }

  private def fromAvgPooling(node: IRElement) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialAveragePooling]
    val kernelW = t.kW
    val kernelH = t.kH
    val strideW = t.dW
    val strideH = t.dH
    val padW = t.padW
    val padH = t.padH
    mkldnn.AvgPooling(kernelW, kernelH, strideW, strideH, padW, padH)
  }

  private def fromLRN(node: IRElement) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialCrossMapLRN]
    val size = t.size
    val alpha = t.alpha
    val beta = t.beta
    val k = t.k
    mkldnn.LRN(size, alpha, beta, k)
  }

  private def fromSbn(node: IRElement) : Module[Float] = {
    val t = node.getOp().asInstanceOf[IRSpatialBatchNormalization[Float]]
    val nOutput = t.nOutput
    val eps = t.eps
    val momentum = t.momentum
    val initWeight = t.initWeight
    val initBias = t.initBias
    val initGradWeight = t.initGradWeight
    val initGradBias = t.initGradBias

    new mkldnn.SpatialBatchNormalization(nOutput, eps, momentum,
      initWeight, initBias, initGradWeight, initGradBias)
  }

//  private def fromLinear(node: IRElement) : Module[Float] = {
//    val t = node.getOp().asInstanceOf[IRLinear]
//    mkldnn.LRN(size, alpha, beta, k)
//  }
}

object IRLayer2Dnn {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRLayer2Dnn = new IRLayer2Dnn
}