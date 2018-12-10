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
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Module, utils}
import com.intel.analytics.bigdl.utils.{MklBlas, Node}
import com.intel.analytics.bigdl.nn.mkldnn

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


private[bigdl] class IRConverter[T: ClassTag](IRgraph: IRGraph[T])(implicit ev: TensorNumeric[T]) {

  /**
   * build to generate BigDL graph
   * @return
   */
  def toGraph() : Graph[T] = {
    if (false) { // utils.Engine.getEngineType() == MklBlas) {
      require(IRToBlas[T].enableConvert(IRgraph.allNodes.toArray),
        "IR graph can not convert to Blas layer")
      toBlasGraph()
    } else {
      require(ev.getType() == FloatType, "Mkldnn engine only supports float data")
      require(IRToDnn[Float].enableConvert(IRgraph.allNodes.toArray.
        asInstanceOf[Array[Node[IRElement[Float]]]]), "IR graph can not convert to Dnn layer")
      toDnnGraph()
    }
  }

  private def toDnnGraph(): Graph[T] = {
    val oldToNew = IRToDnn[Float].convert(IRgraph.allNodes.toArray.
      asInstanceOf[Array[Node[IRElement[Float]]]])
    val inputs = IRgraph.inputs.toArray.map(n =>
                    oldToNew.get(n.asInstanceOf[Node[IRElement[Float]]]).get)
    val outputs = IRgraph.outputs.toArray.map(n =>
                    oldToNew.get(n.asInstanceOf[Node[IRElement[Float]]]).get)

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

    DnnGraph(realInputs, realOutputs,
      IRgraph.variables.asInstanceOf[Option[(Array[Tensor[Float]], Array[Tensor[Float]])]],
      IRgraph.generateBackward).asInstanceOf[Graph[T]]
  }

  private def toBlasGraph(): Graph[T] = {
    val oldToNew = IRToBlas[T].convert(IRgraph.allNodes.toArray)
    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).get)
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).get)

    Graph.dynamic(inputs, outputs, IRgraph.variables, IRgraph.generateBackward)
  }
}
