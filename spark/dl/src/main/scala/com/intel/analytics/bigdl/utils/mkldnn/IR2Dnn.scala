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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Graph, StaticGraph}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node}
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.NodeDef

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.mkldnn

private[mkldnn] class IR2Dnn[T: ClassTag](IRgraph: IRGraph[T])
  (implicit ev: TensorNumeric[T]) extends IRConverter[T](IRgraph) {

  override def mapping(node: IRElement): MklDnnModule = {
    val dnnNode = node.getOp match {
      case "Relu" => mkldnn.ReLU()
      case "DropOut" => mkldnn.Dropout()
      case _ => throw new UnsupportedOperationException(s"Not support layer ${node.getOp}")
    }

    dnnNode.setName(node.getOp)
  }

  override def toGraph(): StaticGraph[T] = {
    val nodes = IRgraph.inputs
    val oldToNew = new util.HashMap[Node[IRElement], Node[MklDnnModule]]()
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
}

object IR2Dnn {
  def apply[T: ClassTag](IRgraph: IRGraph[T])
    (implicit ev: TensorNumeric[T]): IR2Dnn[T] = new IR2Dnn(IRgraph)
}