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
import com.intel.analytics.bigdl.nn.{Graph, StaticGraph, mkldnn}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.tensor.FloatType
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Module, utils}
import com.intel.analytics.bigdl.utils.{MklBlas, Node}
import com.intel.analytics.bigdl.nn.mkldnn

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


private[bigdl] class IRConverter[T: ClassTag](IRgraph: IRGraph[T])(implicit ev: TensorNumeric[T]) {

  /**
   * here to judge whether can convert to dnn graph or blas graph
   * @param pattern
   * @return
   */
  private def enable(pattern: (IRElement[T]) => Boolean) : Boolean = {
    var convert = true
    IRgraph.allNodes.foreach(node => {
      if (!pattern(node.element)) convert = false
    })
    // TODO : log false element name
    convert
  }

  /**
   * build to generate BigDL graph
   * @return
   */
  def toGraph() : Graph[T] = {
    if (utils.Engine.getEngineType() == MklBlas) {
      require(enable(IRLayer2Blas[T].enableConvert), "IR graph can not convert to Blas layer")
      toBlasGraph()
    } else {
      require(ev.getType() == FloatType, "Mkldnn engine only supports float data")
      require(enable(IRLayer2Dnn[T].enableConvert), "IR graph can not convert to Dnn layer")
      toDnnGraph()
    }
  }

  private def cloneNode(oldToNew: mutable.HashMap[Node[IRElement[T]], Node[Module[T]]]) = {
    oldToNew.keySet.toArray.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.contains(nextNodeAndEdge._1)) {
          oldToNew.get(node).get.add(oldToNew.get(nextNodeAndEdge._1).get, nextNodeAndEdge._2)
        }
      })
    })
  }

  private def toDnnGraph(): Graph[T] = {
    val oldToNew = new mutable.HashMap[Node[IRElement[T]], Node[Module[Float]]]()

    IRgraph.allNodes.foreach(node => {
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

    cloneNode(oldToNew)

    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).get)
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).get)

    // add input node for dnn graph
    val realInputs = inputs.map(out => {
      val m = out.element.asInstanceOf[MklDnnLayer]
      val node = new Node(new InputWrapper().asInstanceOf[Module[Float]])
      out.from(node)
      node
    })

    // add output node for graph
    val realOutputs = outputs.map(out => {
      val m = out.element.asInstanceOf[MklDnnLayer]
      val node = new Node(Output(outputLayOut = IRgraph.outputFormats,
        gradOutputLayout = IRgraph.outputFormats).asInstanceOf[Module[Float]])
        out.add(node)
        node
      })

    DnnGraph(realInputs, realOutputs).asInstanceOf[Graph[T]]
  }

  private def toBlasGraph(): Graph[T] = {
    val oldToNew = new mutable.HashMap[Node[IRElement[T]], Node[Module[T]]]()
    IRgraph.allNodes.foreach(node => {
      require(IRLayer2Blas[T].enableConvert(node.element), "")
      val e = IRLayer2Blas[T].convertIRLayer(node.element)
      oldToNew.put(node, new Node(e))
    })

    cloneNode(oldToNew)

    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).get)
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).get)

    Graph.dynamic(inputs, outputs, IRgraph.variables, IRgraph.generateBackward)
  }
}