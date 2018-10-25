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

import java.util.List

import com.intel.analytics.bigdl.nn.{Graph, keras}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.NodeType
import com.intel.analytics.bigdl.nn.mkldnn.NodeType.BigDLNode
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, T}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class GeneralGraph[T: ClassTag, D](
    listNodes: Array[Node[D]],
    _inputs: Seq[String] = null,
    _outputs: Seq[String] = null,
    node: NodeType = BigDLNode)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] with Serializable {

  def inputs(): Seq[String] = _inputs
  def outputs(): Seq[String] = _outputs

  var input_layers = new ArrayBuffer[Node[D]]
  var output_layers = new ArrayBuffer[Node[D]]
  var layer_name_map = T()
  var layer_map = listNodes

  def nodeType(): NodeType = nodeType
  def nodes(): Array[Node[D]] = listNodes

  var graph: Graph[T] = null

  override def updateOutput(input: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("forward not supported")
    }
    graph.updateOutput(input)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported")
    }
    graph.updateGradInput(input, gradOutput)
  }


  def build(): Unit = {
    // first mapping
    makeInputLayers()
    makeOutputLayers()
    // create graph
    // graph = Graph(input_layers, output_layers)
  }

  def makeInputLayers(): Unit = {
    var i = 0
    while (i < layer_map.length) {
      val n = layer_map(i)
      // no previous node
      if (n.prevNodes.length == 0) {
        input_layers.append(n)
      }
      i += 1
    }
  }

  def makeOutputLayers(): Unit = {
    var i = 0
    while (i < layer_map.length) {
      val n = layer_map(i)
      // no next nodes
      if (n.nextNodes.length == 0) {
       output_layers.append(n)
      }
      i += 1
    }
  }
}