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

import com.intel.analytics.bigdl.mkl.{Engine, Memory}
import com.intel.analytics.bigdl.nn.{Graph, StaticGraph}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, HeapData, MklDnnModule, ReorderMemory}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Module, utils}
import com.intel.analytics.bigdl.utils.{MklBlas, Node}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


private[bigdl] class IRConverter[T: ClassTag](IRgraph: IRGraph[T])(implicit ev: TensorNumeric[T]) {

  // here to judge whether can convert to dnn graph
  // just support all layers can be converted to dnn layers
  private def enableConvertDnn(): Boolean = {
    var convert = true
    IRgraph.allNodes.foreach(node => {
      if (!IRLayer2Dnn[T].enableConvert(node.element)) {
        convert = false
      }
    })
    convert
  }

  // here to judge whether can convert to blas graph
  private def enableConvertBlas(): Boolean = {
    var convert = true
    IRgraph.allNodes.foreach(node => {
      if (!IRLayer2Blas[T].enableConvert(node.element)) {
        convert = false
      }
    })
    convert
  }


  def toGraph() : Graph[T] = {
    // build to generate BigDL graph
    if (false) { // utils.Engine.getEngineType() == MklBlas) {
      // convert to Blas
      require(enableConvertBlas(), "this IR graph can not convert to Blas layer")
      toBlasGraph()
    } else {
      // conver to dnn
      require(enableConvertDnn(), "this IR graph can not convert to Blas layer")
      toDnnGraph()
    }
  }

  private def toDnnGraph(): Graph[T] = {
    val allNodes = IRgraph.allNodes
    val oldToNew = new mutable.HashMap[Node[IRElement], Node[Module[Float]]]()
    allNodes.foreach(node => {
      val dnn = if (IRLayer2Dnn[Float].enableConvert(node.element)) {
        new Node(IRLayer2Dnn[Float].convertIRLayer(node.element))
      } else {
        // todo: may be can support non dnn layers
        throw new UnsupportedOperationException(s"can not find ${node.element.getOp()} ")
      }
      oldToNew.put(node, dnn)
    })

    allNodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.contains(nextNodeAndEdge._1)) {
          oldToNew.get(node).get.add(oldToNew.get(nextNodeAndEdge._1).get, nextNodeAndEdge._2)
        }
      })
    })

    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[Float]])
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[Float]])
    // todo: check input formats, if NHWC, then add reorder layer

    // todo: check outputs formats, add output reorder layer
    val realOutputs = IRgraph.outputs.map(out => {
      val shape = out.element.outputShape
      require(shape.length == 2 || shape.length == 4, s"output shape should be 2 or 4 dims," +
        s"but get ${shape.length}, and this IRElement name is ${out.element.getName()}")
      val realOut = if (shape.length == 2) {
        val realOutFormat = new HeapData(shape, Memory.Format.nc)
        val layer = ReorderMemory(inputFormat = null, outputFormat = realOutFormat,
          gradInputFormat = null, gradOutputFomat = realOutFormat)
        layer
      } else if (shape.length == 4) {
        val layer = out.element.getformats() match {
          case "NHWC" =>
            // todo: as mkld nn layer only support nchw format, so here has to do transpose
            // from NHWC to NCHW
            val realOutFormat = new HeapData(out.element.outputShape, Memory.Format.nhwc)
            // val realGradInput = new HeapData(out.element.outputShape, Memory.Format.nchw)
            ReorderMemory(inputFormat = null, outputFormat = realOutFormat,
              gradInputFormat = null, gradOutputFomat = realOutFormat)
          case "NCHW" =>
            val realOutFormat = new HeapData(shape, Memory.Format.nchw)
            ReorderMemory(inputFormat = null, outputFormat = realOutFormat,
              gradInputFormat = null, gradOutputFomat = realOutFormat)
          case _ => throw new UnsupportedOperationException(
            s"not support format ${out.element.getformats()}")
        }
        layer
      }
      val realNode = new Node(realOut.asInstanceOf[Module[Float]])
      oldToNew.get(out).get.add(realNode)
      realNode
    })

    // val outputsFormats = outputs.foreach(n => n.element.asInstanceOf)
    DnnGraph(inputs, realOutputs).asInstanceOf[Graph[T]]
  }

  private def toBlasGraph(): Graph[T] = {
    val allNodes = IRgraph.allNodes
    val oldToNew = new mutable.HashMap[Node[IRElement], Node[Module[T]]]()
    allNodes.foreach(node => {
      require(IRLayer2Blas[T].enableConvert(node.element), "")
      oldToNew.put(node, new Node(IRLayer2Blas[T].convertIRLayer(node.element)))
    })

    allNodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.contains(nextNodeAndEdge._1)) {
          oldToNew.get(node).get.add(oldToNew.get(nextNodeAndEdge._1).get, nextNodeAndEdge._2)
        }
      })
    })

//    allNodes.foreach(node => {
//      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
//        if (oldToNew.contains(nextNodeAndEdge._1)) {
//          val n = oldToNew.get(node).get
//          // todo: use last node
//          n(n.length - 1).add(
//            oldToNew.get(nextNodeAndEdge._1).get(0), nextNodeAndEdge._2)
//        }
//      })
//    })

    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[T]])
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[T]])

    Graph.dynamic(inputs, outputs, IRgraph.variables, IRgraph.generateBackward)
  }
}