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

import breeze.linalg.reverse
import com.intel.analytics.bigdl.nn.{Graph, keras}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, Node, T}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class IRGraph[T: ClassTag](
    val inputs : Seq[Node[IRElement]],
    val outputs : Seq[Node[IRElement]],
    private[bigdl] val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    val generateBackward: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] with Serializable {


  // var input_layers = new ArrayBuffer[Node[IRElement]]
  // var output_layers = new ArrayBuffer[Node[IRElement]]
  var allNodes = new ArrayBuffer[Node[IRElement]]
  // var layer_name_map = T()

  var graph: Graph[T] = null

  // do init to generate all nodes
  init()

  private def init() : Unit = {
    getNodes(inputs, allNodes)
    // todo: some output nodes may not be searched from inputs
    outputs.foreach(node => {
      if (!allNodes.contains(node)) allNodes.append(node)
    })
  }

  private def getNodes(inputs: Seq[Node[IRElement]],
                       nodesBuffer: ArrayBuffer[Node[IRElement]]): Unit = {
    if (inputs.length == 0) return
    inputs.foreach(node => {
      if (!nodesBuffer.contains(node)) nodesBuffer.append(node)
      getNodes(node.nextNodes, nodesBuffer)
    })
  }

  override def updateOutput(input: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("forward not supported, Please build graph first")
    }
    graph.updateOutput(input)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first")
    }
    graph.updateGradInput(input, gradOutput)
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first")
    }
    graph.accGradParameters(input, gradOutput)
  }

  def build(): Unit = {
    // build to generate BigDL graph
    if (false) { // Engine.getEngineType() == MklBlas) {
      // convert to Blas
      // graph = IR2Blas(this).toGraph()
      // graph = IR2Dnn(this).toGraph()
    } else if (IR2Dnn(this).enableConvert()) {
      // conver to dnn
      graph = IR2Dnn(this).toGraph()
    } else {
      // conver to Blas
      graph = IR2Blas(this).toGraph()
    }
  }
}

object IRGraph {
  def apply[T: ClassTag](
    inputs: Seq[Node[IRElement]],
    outputs: Seq[Node[IRElement]],
    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    generateBackward: Boolean = true
  )( implicit ev: TensorNumeric[T]): IRGraph[T] = {
    new IRGraph[T](inputs, outputs, variables, generateBackward)
  }
}