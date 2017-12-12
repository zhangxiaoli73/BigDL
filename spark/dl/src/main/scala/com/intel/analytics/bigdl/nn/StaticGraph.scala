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

import java.util

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.WithoutInput
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, T}

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
  private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)
  (implicit ev: TensorNumeric[T])
  extends Graph[T](_inputs, _outputs, _variables) {

  private val inputCache = new util.HashMap[String, Activity]()
  private val gradOutputCache = new util.HashMap[String, Activity]()
  private val forwardExecution = forwardGraph.topologySort.reverse
  private var backwardExecution: Array[Node[AbstractModule[Activity, Activity, T]]] = _

  buildBackwardGraph()

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = if (node.prevNodes.isEmpty && !node.element.isInstanceOf[WithoutInput]) {
        getInput(node, input)
      } else {
        val prevActivities = node.prevNodesAndEdges.map(n => {
          n._2.fromIndex match {
            case Some(i) => n._1.element.output.toTable.apply[Activity](i)
            case None => n._1.element.output
          }
        })
        if (prevActivities.length == 1) {
          prevActivities.head
        } else {
          T.seq(prevActivities)
        }
      }
      node.element.forward(nodeInput)
      inputCache.put(node.element.getName(), nodeInput)
      i += 1
    }

    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val before = System.nanoTime()
    val gradients = backwardExecution(input, gradOutput, true)
    backwardTime = System.nanoTime() - before
    gradients
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    backwardExecution(input, gradOutput, false)
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecution.length) {
      val curNode = backwardExecution(i)
      curNode.element.accGradParameters(inputCache.get(curNode.element.getName()),
        gradOutputCache.get(curNode.element.getName()))
      i += 1
    }
  }

  override def buildBackwardGraph(): this.type = {
    super.buildBackwardGraph()
    backwardExecution = backwardGraph.topologySort.reverse
    this
  }

  private def backwardExecution(input: Activity, gradOutput: Activity,
    executeBackward: Boolean): Activity = {
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecution.length - 1) {  // do not execute the dummy backward end
      val curNode = backwardExecution(i)
      var curGradOutput : Activity = if (curNode.eq(dummyOutputGrad)) gradOutput else null

      curNode.prevNodesAndEdges.foreach(n => {
        val otherActivity = if (n._1.element.gradInput.isTensor || n._1.nextEdges.length == 1) {
          n._1.element.gradInput
        } else {
          val index = n._1.nextEdges.indexOf(n._2) + 1
          n._1.element.gradInput.toTable.apply[Activity](index)
        }

        n._2.fromIndex match {
          case Some(i) =>
            if (curNode.element.output.isTable && curGradOutput == null) {
              curGradOutput = T()
            }
            val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
            curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
          case None =>
            curGradOutput = accActivity(curGradOutput, otherActivity)
        }
      })

      gradOutputCache.put(curNode.element.getName(), curGradOutput)
      if (!isStopGradient(curNode.element)) {
        if (executeBackward) {
          curNode.element.backward(inputCache.get(curNode.element.getName()), curGradOutput)
        } else {
          curNode.element.updateGradInput(inputCache.get(curNode.element.getName()), curGradOutput)
        }
      } else if (executeBackward) {
        curNode.element.accGradParameters(inputCache.get(curNode.element.getName()), curGradOutput)
      }
      i += 1
    }

    gradInput = if (inputs.length == 1) {
      inputs.head.element.gradInput
    } else {
      T.seq(inputs.map(n => n.element.gradInput))
    }
    gradInput
  }
}
