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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.ControlDependency
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.mkldnn._
import com.intel.analytics.bigdl.utils.{Node, Util}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * A graph container. The modules in the container are connected as a DAG graph.
 *
 * @param _inputs inputs modules, user can feed data into these modules in the forward method
 * @param _outputs output modules
 * @param _variables
 * @tparam T Numeric type. Only support float/double now
 */
class StaticGraph[T: ClassTag](
  private val _inputs : Seq[ModuleNode[T]],
  private val _outputs : Seq[ModuleNode[T]],
  private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
  private val enableExcludeChecking: Boolean = true
)(implicit ev: TensorNumeric[T]) extends Graph[T](_inputs, _outputs, _variables) {
  private val forwardExecution = forwardGraph.topologySort.reverse
  private var backwardExecution: Array[Node[AbstractModule[Activity, Activity, T]]] = _
  private val inputCache = new Array[Activity](forwardExecution.length)
  private var backId2ForwardId: Array[Int] = _
  private var gradOutputCache: Array[Activity] = _

  if (enableExcludeChecking) {
    excludeInvalidLayers(forwardExecution.map {_.element})
  }

  buildBackwardGraph()

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = findInput(node, input)
      inputCache(i) = nodeInput
      node.element.forward(nodeInput)
      i += 1
    }

    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val before = System.nanoTime()
    val gradients = backwardExecution(input, gradOutput, true)
    backwardTime += System.nanoTime() - before
    gradients
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    backwardExecution(input, gradOutput, false)
  }


  override def buildBackwardGraph(): this.type = {
    super.buildBackwardGraph()
    backwardExecution = backwardGraph.topologySort.reverse
    backId2ForwardId = new Array[Int](backwardExecution.length)
    gradOutputCache = new Array[Activity](backwardExecution.length)

    var i = 0
    while(i < backwardExecution.length - 1) {
      var j = 0
      var find = false
      while(j < forwardExecution.length) {
        if (forwardExecution(j).element.getName() == backwardExecution(i).element.getName()) {
          backId2ForwardId(i) = j
          find = true
        }
        j += 1
      }
      require(find, "Cannot find backward layer in forward executions")
      i += 1
    }

    this
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecution.length - 1) {
      val curNode = backwardExecution(i)
      val curInput = inputCache(backId2ForwardId(i))
      curNode.element.accGradParameters(curInput, gradOutputCache(i))
      i += 1
    }
  }

  override def populateModules(): Unit = {
    modules.appendAll(
      forwardGraph.topologySort
        // todo: convert control dep node to edge
        .filterNot(_.element.isInstanceOf[ControlDependency[T]])
        .filter(n => !n.eq(dummyOutput)).map(_.element)
        .reverse
    )
    checkDuplicate()
  }


  private def backwardExecution(input: Activity, gradOutput: Activity,
    executeBackward: Boolean): Activity = {
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecution.length - 1) {  // do not execute the dummy backward end
      val curNode = backwardExecution(i)
      val curGradOutput = findGradOutput(curNode, gradOutput)
      gradOutputCache(i) = curGradOutput
      val curInput = inputCache(backId2ForwardId(i))
      if (!isStopGradient(curNode.element)) {
        if (executeBackward) {
          curNode.element.backward(curInput, curGradOutput)
        } else {
          curNode.element.updateGradInput(curInput, curGradOutput)
        }
      } else if (executeBackward) {
        curNode.element.accGradParameters(curInput, curGradOutput)
      }
      i += 1
    }

    gradInput = fetchModelGradInput()
    gradInput
  }

  private def getNodes(inputs: Seq[Node[Module[T]]],
                       nodesBuffer: ArrayBuffer[Node[Module[T]]]): Unit = {
    if (inputs.length == 0) return
    inputs.foreach(node => {
      if (!nodesBuffer.contains(node)) nodesBuffer.append(node)
      getNodes(node.nextNodes, nodesBuffer)
    })
  }

  def toIRgraph(inputFormats: Int = Memory.Format.nchw,
                outputFormats: Int = Memory.Format.nc) : IRGraph[T] = {
    val allNodes = new ArrayBuffer[Node[Module[T]]]
    getNodes(inputs, allNodes)
    // todo: some output nodes may not be searched from inputs
    outputs.foreach(node => {
      if (!allNodes.contains(node)) allNodes.append(node)
    })

    val oldToNew = new mutable.HashMap[Node[Module[T]], Node[IRElement[T]]]()
    allNodes.foreach(node => {
      val layer = node.element
      val layerName = layer.getClass.getSimpleName
      val cls = Class.forName("com.intel.analytics.bigdl.utils.mkldnn.IR" + layerName)
      val op = ReflectionUtils.reflection(layer, cls).asInstanceOf[IROperate[T]]

      val weightsAndBias =
        if (layer.parameters() != null) layer.getParameters() else (null, null)
      val element = IRElement[T](
        layer.getName(), op, weights = weightsAndBias._1, bias = weightsAndBias._2)

      oldToNew.put(node, new Node(element))
    })

    allNodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.contains(nextNodeAndEdge._1)) {
          oldToNew.get(node).get.add(oldToNew.get(nextNodeAndEdge._1).get, nextNodeAndEdge._2)
        }
      })
    })

    val inputsIR = inputs.toArray.map(n => oldToNew.get(n).get)
    val outputsIR = outputs.toArray.map(n => oldToNew.get(n).get)

    IRGraph(inputsIR, outputsIR, variables,
      inputFormats = inputFormats, outputFormats = Memory.Format.nchw)
  }
}
