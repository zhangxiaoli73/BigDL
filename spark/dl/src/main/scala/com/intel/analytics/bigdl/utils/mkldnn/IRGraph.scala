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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, Node, T}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class IRGraph[T: ClassTag](
    val inputs : Seq[Node[IRElement]],
    val outputs : Seq[Node[IRElement]],
    val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    val generateBackward: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] with Serializable {


  var allNodes = new ArrayBuffer[Node[IRElement]]
  private[bigdl] var graph: Graph[T] = null

  /* init should do:
   * 1. find all nodes and 4 dimention formats
   * 2. infer input shape & output shape for all IRelement
   */
  init()

  private def init() : Unit = {
    getNodes(inputs, allNodes)
    // todo: some output nodes may not be searched from inputs
    outputs.foreach(node => {
      if (!allNodes.contains(node)) allNodes.append(node)
    })

    // check formats
    val inputFormats = DataFormat(checkModelFormats())

    // add format type for all nodes element
    allNodes.foreach(node => {
      node.element.setFormats(inputFormats.value)
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

  private def checkModelFormats() : String = {
    var formats = new ArrayBuffer[String]()
    allNodes.foreach(node => {
      // todo: data format not match should be also false
      val f = node.element.getformats()
      if (f != "" && f != null) formats.append(f)
    })
    require(formats.distinct.length == 1, "wrong format transfer")
    formats(0)
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
    new IRConverter[T](this).toGraph()
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