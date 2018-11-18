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

class IRToDnn extends ConvertBase[IRElement[Float], Module[Float]] {

  // converter function mappings
  private val IR2DnnMap = new mutable.HashMap[String, (IRElement[Float]) => Module[Float]]
  mapInit()

  private def mapInit(): Unit = {
    IR2DnnMap("IRSpatialConvolution") = fromConv
    IR2DnnMap("IRSpatialMaxPooling") = fromMaxPooling
    IR2DnnMap("IRSpatialAveragePooling") = fromAvgPooling
    IR2DnnMap("IRSpatialBatchNormalization") = fromSbn
    IR2DnnMap("IRLinear") = fromLinear
    IR2DnnMap("IRReLU") = fromReLU
    IR2DnnMap("IRLogSoftMax") = fromLogSoftMax
    IR2DnnMap("IRSpatialShareConvolution") = fromConv
  }

  override def enableConvertLayer(layer: IRElement[Float]): Boolean = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name)) return true
    val className = "com.intel.analytics.bigdl.nn.mkldnn." + name.substring(2)
    val cls = ReflectUtils.classFound(className)
    if ( cls != null) true
    else false
  }

  override def convertLayer(layer: IRElement[Float]) : Module[Float] = {
    val name = layer.getOp().name
    if (IR2DnnMap.contains(name)) {
      val dnn = IR2DnnMap(name)(layer)
      if (dnn.parameters() != null) {
        val params = dnn.getParameters()
        val params2 = layer.getParameters()
        if (params2._1 != null) params._1.copy(params2._1)
        if (params2._2 != null) params._2.copy(params2._2)
      }
      if (layer.getName != "") dnn.setName(layer.name)
      dnn
    } else {
      val cls = Class.forName("com.intel.analytics.bigdl.nn.mkldnn." + name.substring(2))
      ReflectUtils.reflectFromIR(layer, cls)
    }
  }

  override def enableConvert(allNodes: Array[Node[IRElement[Float]]]) : Boolean = {
    var convert = true
    allNodes.foreach(node => {
      val op = node.element.getOp()
      if (op.isInstanceOf[IRReshape[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        // support pattern "reshape -> linear"
      } else if (op.isInstanceOf[IRView[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        // support pattern "view -> linear"
      } else if (!enableConvertLayer(node.element)) {
        convert = false
      }
    })
    // TODO : log false element name
    convert
  }

  override def convert(allNodes: Array[Node[IRElement[Float]]])
    : mutable.HashMap[Node[IRElement[Float]], Node[Module[Float]]] = {
    val oldToNew = new mutable.HashMap[Node[IRElement[Float]], Node[Module[Float]]]()
    allNodes.foreach(node => {
      val op = node.element.getOp()
      val dnn = if (op.isInstanceOf[IRReshape[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        new Node(mkldnn.Identity[Float]().asInstanceOf[Module[Float]])
      } else if (op.isInstanceOf[IRView[Float]] && node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().isInstanceOf[IRLinear[Float]]) {
        new Node(mkldnn.Identity[Float]().asInstanceOf[Module[Float]])
      } else {
        if (enableConvertLayer(node.element)) {
          new Node(convertLayer(node.element))
        } else {
          // todo: may be can support non dnn layers
          throw new UnsupportedOperationException(s"can not find ${node.element.getOp()} ")
        }
      }
      oldToNew.put(node, dnn)
    })
    cloneNode(oldToNew)
    oldToNew
  }

  private def fromReLU(node: IRElement[Float]) : Module[Float] = mkldnn.ReLU()

  private def fromLogSoftMax(node: IRElement[Float]) : Module[Float] = {
    BigDL2DnnWrapper(nn.LogSoftMax[Float]())
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

    val extraParams = layer.getExtraParameter()
    if (t.runningMean != null) extraParams(0).copy(t.runningMean.toTensor[Float])
    if (t.runningVar != null) extraParams(1).copy(t.runningVar.toTensor[Float])

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

    if (node.name == "fc1") {
      val tmp = 0
    }
    val params = node.getParameters()
    val params2 = layer.getParameters()
    if (params._1 != null) params2._1.copy(params._1)
    if (params._2 != null) params2._2.copy(params._2)

    layer
  }
}

object IRToDnn {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRToDnn = new IRToDnn
}