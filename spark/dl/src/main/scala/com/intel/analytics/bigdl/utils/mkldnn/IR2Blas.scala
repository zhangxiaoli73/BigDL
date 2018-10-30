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

import java.util

import com.intel.analytics.bigdl.nn.{Graph, StaticGraph}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.reflect.ClassTag


class IR2Blas[T: ClassTag](IRgraph: IRGraph[T])
  (implicit ev: TensorNumeric[T]) extends IRConverter[T](IRgraph) {

  override def mapping(node: IRElement): Module[T] = {
    import com.intel.analytics.bigdl.nn
    nn.ReLU().asInstanceOf[Module[T]]
  }

  override def toGraph(): Graph[T] = {
    val nodes = IRgraph.inputs
    val oldToNew = new util.HashMap[Node[IRElement], Node[Module[T]]]()
    nodes.foreach(node => {
      val dnn = new Node(mapping(node.element))
      oldToNew.put(node, dnn)
    })

    nodes.foreach(node => {
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

  override def enableConvert(): Boolean = {
   true
  }
}

object IR2Blas {
  def apply[T: ClassTag](IRgraph: IRGraph[T])
                        (implicit ev: TensorNumeric[T]): IR2Blas[T] = new IR2Blas(IRgraph)
}