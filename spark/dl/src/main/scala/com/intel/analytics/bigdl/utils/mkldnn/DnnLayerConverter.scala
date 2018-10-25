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
import java.util.List

import breeze.linalg.reverse
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}
import org.tensorflow.framework.NodeDef

import scala.collection.mutable
import scala.reflect.ClassTag


abstract class DnnLayerConverter[T: ClassTag, D: ClassTag](defGraph: GeneralGraph[T, D])
                                                          (implicit ev: TensorNumeric[T]) {

  def mapping(node: D): MklDnnModule

  def toDnnGraph(): StaticGraph[T]

  def cloneGraph(): DirectedGraph[T] = {
    val oldToNew = new util.HashMap[Node[D], Node[MklDnnModule]]()
    val nodes = defGraph.nodes()
    nodes.foreach(node => {
      oldToNew.put(node, new Node[MklDnnModule](mkldnn.Identity()))
    })
    // Keep the order in the nextNodes array and prevNodes array of the current node.
    // As we go through all node in bfs from source, the prevNodes order can be preserved.
    // For each node, we iterate and add their nextNodes, the nextNodes order can also be preserved.
    nodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.containsKey(nextNodeAndEdge._1)) {
          oldToNew.get(node).add(oldToNew.get(nextNodeAndEdge._1), nextNodeAndEdge._2)
        }
      })
    })
    oldToNew
  }
}
