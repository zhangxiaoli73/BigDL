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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import spire.macros.Auto.scala

private[mkldnn] class IR2Dnn[T: ClassTag](IRgraph: IRGraph[T])
  (implicit ev: TensorNumeric[T]) extends IRConverter[T](IRgraph) {

  private def toConv(node: IRElement) : MklDnnModule = {
    val t = node.getAttrMap()
    val nInputPlane = t.getOrElse("nInputPlane", null).asInstanceOf[Int]
    val nOutputPlane = t.getOrElse("nOutputPlane", null).asInstanceOf[Int]
    val kernelW = t.getOrElse("kernelW", null).asInstanceOf[Int]
    val kernelH = t.getOrElse("kernelH", null).asInstanceOf[Int]
    val strideW = t.getOrElse("strideW", null).asInstanceOf[Int]
    val strideH = t.getOrElse("strideH", null).asInstanceOf[Int]
    val padW = t.getOrElse("pW", null).asInstanceOf[Int]
    val padH = t.getOrElse("pH", null).asInstanceOf[Int]
    val initWeight = t.getOrElse("weights", null).asInstanceOf[Tensor[Float]]
    val initBias = t.getOrElse("bias", null).asInstanceOf[Tensor[Float]]
    val initGradWeight = t.getOrElse("gradWeights", null).asInstanceOf[Tensor[Float]]
    val initGradBias = t.getOrElse("gradBias", null).asInstanceOf[Tensor[Float]]
    val format = t.getOrElse("format", null).asInstanceOf[DataFormat]

    mkldnn.SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, initWeight = initWeight, initBias = initBias,
      initGradWeight = initGradWeight, initGradBias = initGradBias, format = format)
  }

  private def toMaxPooling(node: IRElement) : MklDnnModule = {
    val t = node.getAttrMap()
    val format = t.getOrElse("data_format", null).asInstanceOf[DataFormat]
    val strideList = t.getOrElse("strides", null).asInstanceOf[ArrayBuffer[Int]]
    val kernelList = t.getOrElse("ksize", null).asInstanceOf[ArrayBuffer[Int]]
    val (strideH, strideW, ksizeH, ksizeW) = format match {
      case DataFormat.NHWC =>
        // todo: not support
        require(strideList(3) == 1, s"not support strides on depth")
        (strideList(1), strideList(2), kernelList(1), kernelList(2))
      case DataFormat.NCHW =>
        require(strideList(1) == 1, s"not support strides on depth")
        (strideList(2), strideList(3), kernelList(2), kernelList(3))
      case _ =>
        throw new IllegalArgumentException(s"not supported data format: $format")
    }

    val padding = t.get("padding")
    val (pW, pH) =
      if (padding == "SAME") {
        (-1, -1)
      } else {
        (0, 0)
      }
    mkldnn.MaxPooling(ksizeW, ksizeH, strideW, strideH, pW, pH)
  }

  override def mapping(node: IRElement): Module[T] = {
    val dnnNode = node.getOp match {
      case "Placeholder" => mkldnn.Identity()
      case "Relu" => mkldnn.ReLU()
      case "DropOut" => mkldnn.Dropout()
      case "Identity" => mkldnn.Identity()
      case "SpatialConvolution" => toConv(node)
      case "MaxPool" => toMaxPooling(node)
        // todo: handle squeeze
      case "Squeeze" => mkldnn.Identity()
      case _ => throw new UnsupportedOperationException(s"Not support layer ${node.getOp}")
    }

    dnnNode.asInstanceOf[Module[T]].setName(node.getName())
  }

  def printTest(nodes: Seq[Node[IRElement]]): Unit = {
    if (nodes.length == 0) {
      return
    }
    nodes.foreach(node => {
      println(node.element.getOp())
      printTest(node.nextNodes)
    })
  }
  override def toGraph(): StaticGraph[T] = {
    val nodes = IRgraph.inputs
    printTest(nodes)

    val allnodes = IRgraph.getAllNodes()
    val oldToNew = new util.HashMap[Node[IRElement], Node[MklDnnModule]]()
    allnodes.foreach(node => {
      val dnn = new Node(mapping(node.element))
      oldToNew.put(node, dnn.asInstanceOf[Node[MklDnnModule]])
    })

    allnodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.containsKey(nextNodeAndEdge._1)) {
          oldToNew.get(node).add(oldToNew.get(nextNodeAndEdge._1), nextNodeAndEdge._2)
        }
      })
    })

    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[T]])
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[T]])

    new StaticGraph[T](inputs, outputs)
  }

  val map = new mutable.HashMap[String, MklDnnModule]

  override def enableConvert(): Boolean = {
    var convert = true
    val nodes = IRgraph.inputs
    nodes.foreach(node => {
      // todo: data format not match should be also false
      if (!map.contains(node.element.getOp())) {
        convert = false
      }
    })
    convert
  }
}

object IR2Dnn {
  def apply[T: ClassTag](IRgraph: IRGraph[T])
    (implicit ev: TensorNumeric[T]): IR2Dnn[T] = new IR2Dnn(IRgraph)
}