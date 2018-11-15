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
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Module, utils}
import com.intel.analytics.bigdl.utils.{MklBlas, Node}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


private[bigdl] class IRConverter[T: ClassTag](IRgraph: IRGraph[T])(implicit ev: TensorNumeric[T]) {

  // here to judge whether can convert to dnn graph
  // just support all layers can be converted to dnn layers
  private def enableToDnn(): Boolean = {
    var convert = true
    IRgraph.allNodes.foreach(node => {
      if (!IRLayer2Dnn[Float].enableConvert(node.element.asInstanceOf[IRElement[Float]])) {
        convert = false
      }
    })
    convert
  }

  // here to judge whether can convert to blas graph
  private def enableToBlas(): Boolean = {
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
    if (utils.Engine.getEngineType() == MklBlas) {
      // convert to Blas
      require(enableToBlas(), "this IR graph can not convert to Blas layer")
      toBlasGraph()
    } else {
      // conver to dnn
      // require(enableToDnn(), "this IR graph can not convert to Blas layer")
      toDnnGraph()
    }
  }

  private[bigdl] def toDnnGraph(): Graph[T] = {
    import com.intel.analytics.bigdl.nn.mkldnn
    val allNodes = IRgraph.allNodes
    val oldToNew = new mutable.HashMap[Node[IRElement[T]], Node[Module[Float]]]()

    allNodes.foreach(node => {
      val op = node.element.getOp().asInstanceOf[IROperate[Float]]
      val dnn = if (op.isInstanceOf[IRReshape[Float]] &&
        node.nextNodes.length == 1 &&
        node.nextNodes(0).element.getOp().asInstanceOf[IROperate[Float]]
          .isInstanceOf[IRLinear[Float]]) {
        new Node(mkldnn.Identity().asInstanceOf[Module[Float]])
      } else {
        if (IRLayer2Dnn[Float].enableConvert(node.element.asInstanceOf[IRElement[Float]])) {
          val e = IRLayer2Dnn[Float].convertIRLayer(node.element.asInstanceOf[IRElement[Float]])
          new Node(e)
        } else {
          // todo: may be can support non dnn layers
          throw new UnsupportedOperationException(s"can not find ${node.element.getOp()} ")
        }
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

    val inputs = IRgraph.inputs.toArray.map(n =>
      oldToNew.get(n).get.asInstanceOf[ModuleNode[Float]])
    val outputs = IRgraph.outputs.toArray.map(n =>
      oldToNew.get(n).get.asInstanceOf[ModuleNode[Float]])

    val realInputs = inputs.map(out => {
      val m = out.element.asInstanceOf[MklDnnLayer]
      val node = new Node(new InputWrapper().asInstanceOf[Module[Float]])
      out.from(node)
      node
    })

    val realOutputs = outputs.map(out => {
      val m = out.element.asInstanceOf[MklDnnLayer]
      val node = new Node(Output(outputLayOut = IRgraph.outputFormats,
        gradOutputLayout = IRgraph.outputFormats).asInstanceOf[Module[Float]])
        out.add(node)
        node
      })

    DnnGraph(realInputs, realOutputs).asInstanceOf[Graph[T]]
  }

  private[bigdl] def toBlasGraph(): Graph[T] = {
    val allNodes = IRgraph.allNodes
    val oldToNew = new mutable.HashMap[Node[IRElement[T]], Node[Module[T]]]()
    allNodes.foreach(node => {
      require(IRLayer2Blas[T].enableConvert(node.element), "")
      val e = IRLayer2Blas[T].convertIRLayer(node.element)
      oldToNew.put(node, new Node(e))
    })

    allNodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.contains(nextNodeAndEdge._1)) {
          oldToNew.get(node).get.add(oldToNew.get(nextNodeAndEdge._1).get, nextNodeAndEdge._2)
        }
      })
    })

    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).get.asInstanceOf[ModuleNode[T]])
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).get.asInstanceOf[ModuleNode[T]])

    Graph.dynamic(inputs, outputs, IRgraph.variables, IRgraph.generateBackward)
  }
}