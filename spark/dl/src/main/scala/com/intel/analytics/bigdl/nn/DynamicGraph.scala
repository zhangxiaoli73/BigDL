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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.ops.ControlOps
import com.intel.analytics.bigdl.nn.tf.{ControlDependency, WithoutInput}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable
import scala.reflect.ClassTag

class DynamicGraph[T: ClassTag](
  private val _inputs : Seq[ModuleNode[T]],
  private val _outputs : Seq[ModuleNode[T]],
  private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
  val generateBackward: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends Graph[T](_inputs, _outputs, _variables) {

  buildBackwardGraph()

  override def updateOutput(input: Activity): Activity = {
    forwardScheduler.reset()
    while (!forwardScheduler.isFinished()) {
      val node = forwardScheduler.fetch()
      val nodeInput = findInput(node, input)
      node.element.forward(nodeInput)
      forwardScheduler.schedule(node)
    }

    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val before = System.nanoTime()
    val result = backwardExecution(input, gradOutput, true)
    backwardTime = System.nanoTime() - before
    result
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    backwardExecution(input, gradOutput, false)
  }

  private def backwardExecution(input: Activity, gradOutput: Activity, isBackward: Boolean)
  : Activity = {
    if (!generateBackward) return null
    backwardScheduler.reset()
    while (!backwardScheduler.isFinished()) {
      val curNode = backwardScheduler.fetch()
      val curGradOutput = findGradOutput(curNode, gradOutput)
      if (!isStopGradient(curNode.element)) {
        if (isBackward) {
          curNode.element.backward(inputCache(curNode.element.getName()), curGradOutput)
        } else {
          curNode.element.updateGradInput(inputCache(curNode.element.getName()), curGradOutput)
        }
      } else if (isBackward) {
        curNode.element.accGradParameters(inputCache(curNode.element.getName()), curGradOutput)
      }
      backwardScheduler.schedule(curNode)
    }

    gradInput = fetchModelGradInput()
    gradInput
  }

  private val forwardScheduler = new Scheduler(
    forwardNodes.filter(_.prevNodes.length == 0),
    Seq(dummyOutput),
    forwardNodes.map(_.element.getName()).toSet
  )
  private var backwardScheduler : Scheduler[T] = _

  /**
   * Generate backward graph and apply the stopGrad
   */
  override private[bigdl] def buildBackwardGraph(): this.type = {
    if (!generateBackward) return this

    forwardNodes.foreach(n => require(!n.element.isInstanceOf[ControlOps[_]],
      "Not suppot generate back graph with control ops node"))

    super.buildBackwardGraph()
    val forwardNodeNames = forwardNodes.map(_.element.getName()).toSet
    val executableNodes = backwardGraph.DFS.map(_.element.getName())
      .filter(forwardNodeNames.contains(_)).toSet

    val inputNames = inputs.map(_.element.getName()).toSet
    val backwardTargets = backwardNodes
      .filter(n => (n.element.parameters() != null && n.element.parameters()._1.length != 0)
        || inputNames.contains(n.element.getName()))

    backwardScheduler = new Scheduler[T](
      Seq(dummyOutputGrad),
      backwardTargets,
      executableNodes
    )
    clearState()
    this
  }
}
