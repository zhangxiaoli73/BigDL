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

import breeze.linalg.Axis._1
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.nn.tf.{ControlDependency, WithoutInput}
import com.intel.analytics.bigdl.utils.{Node, T}

import scala.reflect.ClassTag


class DnnGraph[T: ClassTag](
    private val _inputs : Seq[ModuleNode[T]],
    private val _outputs : Seq[ModuleNode[T]],
    private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
    private val enableExcludeChecking: Boolean = true
  )(implicit ev: TensorNumeric[T])
  extends StaticGraph[T](_inputs, _outputs, _variables, enableExcludeChecking) {
  private var forwardExecution: Array[Node[AbstractModule[Activity, Activity, T]]] = _
  private var backwardExecution: Array[Node[AbstractModule[Activity, Activity, T]]] = _
  private var inputCache: Array[Activity] = _
  private var backId2ForwardId: Array[Int] = _
  private var gradOutputCache: Array[Activity] = _
  private var inputFormats: Array[MemoryData] = _
  private var gradOutputFormats: Array[MemoryData] = _

  if (enableExcludeChecking) {
    excludeInvalidLayers(forwardExecution.map {_.element})
  }

  buildBackwardGraph()

  toDnn()

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = findDnnInput(node, input, this.inputFormats)
      // todo: use input for backward???
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
    forwardExecution = forwardGraph.topologySort.reverse
    backwardExecution = backwardGraph.topologySort.reverse
    backId2ForwardId = new Array[Int](backwardExecution.length)
    gradOutputCache = new Array[Activity](backwardExecution.length)
    inputCache = new Array[Activity](forwardExecution.length)

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

  private def findDnnInput(node: ModuleNode[T], input: Activity,
                                  inputFormats: Array[MemoryData])
    : Activity = {
    if (node.element.isInstanceOf[WithoutInput]) return null

    val realInputFormats = node.element.asInstanceOf[MklDnnModule].inputFormats()

    val (nodeInput, lastOutputFormats) = if (node.prevNodes.isEmpty) {
      (getInput(node, input), inputFormats)
    } else {
      val prevActivitiesAndFormats = node.prevNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .map(n => {
          val out = n._2.fromIndex match {
            case Some(i) =>
              if (n._1.element.output == null || (i == 1 && n._1.element.output.isTensor)) {
                n._1.element.output
              } else {
                n._1.element.output.toTable.apply[Activity](i)
              }
            case None => n._1.element.output
          }
          (out, n._1.element.asInstanceOf[MklDnnModule].outputFormats())
        })

      if (prevActivitiesAndFormats.length == 1) {
        prevActivitiesAndFormats.head
      } else {
        (T.seq(prevActivitiesAndFormats.map(m => m._1)),
          prevActivitiesAndFormats.map(m => m._2).toArray.flatMap(_.toSeq))
      }
    }

    // todo: reorder register only once
    lastOutputFormats.zip(realInputFormats).foreach {
      case (o, i) => reorderManager.register(o, i)
    }
    val realInput = reorderManager.infer(
      lastOutputFormats,
      realInputFormats,
      nodeInput
    )

    realInput
  }

  private def findDnnGradOutput(curNode: ModuleNode[T], gradOutput: Activity,
                                gradOutputFormats: Array[MemoryData],
                                isAcc: Boolean = false): Activity = {
    var curGradOutput : Activity = if (curNode.eq(dummyOutputGrad)) gradOutput else null

    val realGradOutputFormats = if (isAcc) {
      curNode.element.asInstanceOf[MklDnnModule].gradOutputFormats()
    } else {
      curNode.element.asInstanceOf[MklDnnModule].gradOutputWeightFormats()
    }

      curNode.prevNodesAndEdges.filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
      .foreach(n => {
        val otherActivity = if (n._1.element.gradInput.isTensor || n._1.nextEdges.length == 1) {
          n._1.element.gradInput
        } else {
          val index = n._1.nextEdges.indexOf(n._2) + 1
          n._1.element.gradInput.toTable.apply[Activity](index)
        }

        n._2.fromIndex match {
          case Some(i) =>
            if (i == 1 && curNode.element.output.isTensor) {
              curGradOutput = accActivity(curGradOutput, realGradOutputFormats,
                otherActivity, n._1.element.asInstanceOf[MklDnnModule].gradInputFormats())
            } else {
              if (curNode.element.output.isTable && curGradOutput == null) {
                curGradOutput = T()
              }
              val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
              curGradOutput.toTable(i) = accActivity(curActivity, realGradOutputFormats,
                otherActivity, n._1.element.asInstanceOf[MklDnnModule].gradInputFormats())
            }
          case None =>
            curGradOutput = accActivity(curGradOutput, realGradOutputFormats,
              otherActivity, n._1.element.asInstanceOf[MklDnnModule].gradInputFormats())
        }
      })

    if (curNode.element.output.isTable) {
      addZeroTensorToMissingGradOutput(curNode.element.output.toTable, curGradOutput.toTable)
    }

    curGradOutput
  }

  private def accActivity(activity: Activity, formats: Array[MemoryData],
                          other: Activity, otherFormats: Array[MemoryData]): Activity = {
    if (activity == null) {
      other
    } else {
      if (other.isTensor) {
        require(activity.isTensor, "Cannot add a table to a tensor")
        // for same layout, no reorder
        if (formats(0).layout == otherFormats(0).layout) {
          activity.toTensor[T].add(other.toTensor[T])
        } else {
          // todo: register only once
          reorderManager.register(formats(0), otherFormats(0))
          activity.toTensor[T].add(reorderManager.infer(
            formats,
            otherFormats,
            other).toTensor[T])
        }
      } else {
        // if 'activity' and 'other' are both table, we need to merge 'other' to 'activity'
        // if 'other' and 'activity' both contains the index, update 'activity' by sum
        // if 'other' contains the index while 'activity' does not,
        // just insert the corresponding tensor of 'other' to 'activity'
        // todo: Please attention all elements in one Table should be same formats
        val actTable = activity.toTable
        val otherTable = other.toTable
          otherTable.keySet.foreach(index => {
            if (actTable.contains(index)) {
              accActivity(actTable[Activity](index),
                reorderManager.infer(formats, otherFormats, otherTable[Activity](index)))
            } else {
              actTable.insert(index.asInstanceOf[Int],
                reorderManager.infer(formats, otherFormats, otherTable(index)))
            }
          })

        actTable
      }
    }
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecution.length - 1) {
      val curNode = backwardExecution(i)
      // todo: use input in forward???
      val curInput = inputCache(backId2ForwardId(i))
      val curGradOutput = findDnnGradOutput(curNode, gradOutput, this.gradOutputFormats, true)
      // curNode.element.accGradParameters(curInput, gradOutputCache(i))
      curNode.element.accGradParameters(curInput, curGradOutput)
      curNode.element.asyncGradient()
      i += 1
    }
  }

  private def backwardExecution(input: Activity, gradOutput: Activity,
                                executeBackward: Boolean): Activity = {
    dummyOutputGrad.element.gradInput = gradOutput

    var i = 0
    while (i < backwardExecution.length - 1) {  // do not execute the dummy backward end
      val curNode = backwardExecution(i)
      val curGradOutput = findDnnGradOutput(curNode, gradOutput, this.gradOutputFormats)
      gradOutputCache(i) = curGradOutput
      // todo: use input in forward???
      val curInput = inputCache(backId2ForwardId(i))
      // val curInput = findDnnInput(curNode, input, this.inputFormats)
      // inputCache(i) = curInput

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

  @transient protected lazy val reorderManager = new ReorderManager()

  val fuseConvBn = System.getProperty("bigdl.mkldnn.fusion.convbn", "false").toBoolean
  val fuseBnRelu = System.getProperty("bigdl.mkldnn.fusion.bnrelu", "false").toBoolean
  val fuseConvRelu = System.getProperty("bigdl.mkldnn.fusion.convrelu", "false").toBoolean
  val fuseConvSum = System.getProperty("bigdl.mkldnn.fusion.convsum", "false").toBoolean

  private def fuseModule(node: Node[AbstractModule[Activity, Activity, T]]) = {
    // fuse relu & conv(false)/bn(false)
    if (node.element.isInstanceOf[mkldnn.ReLU]) {
      node.prevNodes.foreach(n => {
        n.element match {
          case conv
            if (conv.isInstanceOf[mkldnn.SpatialConvolution] && fuseConvRelu) =>
            val element = n.element.asInstanceOf[mkldnn.SpatialConvolution]
            if (!element.relu) {
              element.setReLU(true)
              node.element =
                mkldnn.Identity().asInstanceOf[AbstractModule[Activity, Activity, T]]
            }
          case bn
            if (bn.isInstanceOf[mkldnn.SpatialBatchNormalization] && fuseBnRelu) =>
            val element = n.element.asInstanceOf[mkldnn.SpatialBatchNormalization]
            if (!element.relu) {
              element.setReLU(true)
              node.element =
                mkldnn.Identity().asInstanceOf[AbstractModule[Activity, Activity, T]]
            }
          case _ => null
        }})
    }

    // fuse bn & conv(false)
    if (node.element.isInstanceOf[mkldnn.SpatialBatchNormalization]) {
      node.prevNodes.foreach(n => {
        n.element match {
          case s
            if (s.isInstanceOf[mkldnn.SpatialConvolution] && fuseConvBn) =>
            val element = n.element.asInstanceOf[mkldnn.SpatialConvolution]
            if (!element.relu && !element.batchNorm) {
              fusionConvBn(element, node.element.asInstanceOf[mkldnn.SpatialBatchNormalization])
              node.element =
                mkldnn.Identity().asInstanceOf[AbstractModule[Activity, Activity, T]]
            }
          case _ => null
        }})
    }
    // TODO: conv & CAddTable
    if (node.element.isInstanceOf[mkldnn.CAddTable]) {
      node.prevNodes.foreach(n => {
      })
    }
  }

  private def fusionConvBn(conv: mkldnn.SpatialConvolution,
                           bn: mkldnn.SpatialBatchNormalization): Unit = {

    val originVar = Tensor[Float].resize(bn.runningVariance.size()).copy(bn.runningVariance.dense)
    val originMean = Tensor[Float].resize(bn.runningMean.size()).copy(bn.runningMean.dense)

    val convWeight = Tensor[Float].resize(conv.weight.size()).copy(conv.weight.dense)
    val convBias = Tensor[Float].resize(conv.bias.size()).copy(conv.bias.dense)

    (0 until bn.nOutput).foreach { j =>
      val variance = originVar.storage().array()(j + originVar.storageOffset() - 1)
      val base = Math.sqrt(variance.asInstanceOf[Float] + bn.eps).toFloat
      require(base != 0.0, s"the eps of ${bn.getName()} should be more than 0")

      val weight = if (conv.nGroup == 1) {
        convWeight.select(1, j + 1)
      } else {
        convWeight.select(2, j + 1)
      }
      weight.div(base)

      val bias = convBias.storage().array()(j)
      val mean = originMean.storage().array()(j)
      convBias.storage().array()(j) = (bias - mean) / base
    }

    conv.weight.copy(convWeight)
    conv.bias.copy(convBias)
  }

  private def toDnn(): AbstractModule[Activity, Activity, T] = {
    forwardExecution.foreach(nodes => {
      if (!nodes.element.isInstanceOf[MklDnnModule]) {
        nodes.element = nodes.element.toDnnModule()
      }
      fuseModule(nodes)
    })
    backwardExecution.zip(backId2ForwardId).foreach { case (nodes, num) =>
      if (num >= 0) {
        nodes.element = forwardExecution(num).element
      } else if (!nodes.element.isInstanceOf[MklDnnModule]) {
        nodes.element = nodes.element.toDnnModule()
      }
    }
    this
  }

  def compile(phase: Phase, formats: Array[MemoryData]) : Unit = {
    initPrimitives(phase, new MklDnnRuntime(), formats)
  }

  // todo: why just input formats, no gradOutput formats?
  private def initPrimitives(phase: Phase, runtime: MklDnnRuntime, formats: Array[MemoryData])
  : Unit = {
    setRuntime(runtime, phase)
    val outputFormats = initFwdPrimitives(formats, phase)._2
    inputFormats = formats
    gradOutputFormats = outputFormats
    if (phase == Phase.TrainingPhase) {
      initBwdPrimitives(outputFormats, phase)
      initGradWPrimitives(outputFormats, phase)
    }
  }

  private def setRuntime(runtime: MklDnnRuntime, phase: Phase): Unit = {
    reorderManager.setRuntime(runtime)
    forwardExecution.foreach(m => m.element.asInstanceOf[MklDnnModule].setRuntime(runtime))
    if (phase == Phase.TrainingPhase) {
      backwardExecution.foreach(m => m.element.asInstanceOf[MklDnnModule].setRuntime(runtime))
    }
  }

  private def findInputFormats(node: ModuleNode[T], inputs: Array[MemoryData])
    : Array[MemoryData] = {
    if (node.prevNodes.isEmpty) {
      inputs
    } else {
      val prevFormats = node.prevNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .map(n => n._1.element.asInstanceOf[MklDnnModule].outputFormats()).toArray
      prevFormats.flatMap(n => n.toSeq)
    }
  }

  private def findGradOutputFormats(node: ModuleNode[T], inputs: Array[MemoryData])
    : Array[MemoryData] = {
    if (node.prevNodes.isEmpty) {
      inputs
    } else {
      val prevFormats = node.prevNodesAndEdges
        .filterNot(n => n._1.element.isInstanceOf[ControlDependency[T]])
        .map(n => n._1.element.asInstanceOf[MklDnnModule].gradInputFormats()).toArray
      prevFormats.flatMap(n => n.toSeq)
    }
  }

  // init forward nodes
  private def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase)
  : (Array[MemoryData], Array[MemoryData]) = {
    var lastOutputFormats = inputs
    var firstRealInputFormats: Array[MemoryData] = null
    for (i <- 0 until forwardExecution.length) {
      val m = forwardExecution(i)
      val (realInputFormats, outputFormats) = m.element.asInstanceOf[MklDnnModule]
        .initFwdPrimitives(findInputFormats(m, lastOutputFormats), phase)
//      lastOutputFormats.zip(realInputFormats).foreach {
//        case (o, i) => reorderManager.register(o, i)
//      }
      if (i == 0) firstRealInputFormats = realInputFormats
      lastOutputFormats = outputFormats
    }
    (firstRealInputFormats, lastOutputFormats)
  }

  // init backward nodes
  def initBwdPrimitives(grads: Array[MemoryData], phase: Phase)
  : (Array[MemoryData], Array[MemoryData]) = {
    var lastGradInputFormats = grads
    var firstRealGradOutputFormats: Array[MemoryData] = null
    for (i <- 0 until backwardExecution.length ) {
      val m = backwardExecution(i)
      val (realGradOutput, gradInputFomrats) = m.element.asInstanceOf[MklDnnModule].
        initBwdPrimitives(findGradOutputFormats(m, lastGradInputFormats), phase)
//      lastGradInputFormats.zip(realGradOutput).foreach {
//        case (gi, go) => reorderManager.register(gi, go)
//      }
      if (i == 0) firstRealGradOutputFormats = realGradOutput
      lastGradInputFormats = gradInputFomrats
    }
    (firstRealGradOutputFormats, lastGradInputFormats)
  }

  // init gradoutput
  def initGradWPrimitives(grads: Array[MemoryData], phase: Phase):
  Array[MemoryData] = {
    var lastGradInputFormats = grads
    var firstRealGradOutputFormats: Array[MemoryData] = null
    for (i <- 0 until backwardExecution.length) {
      val m = backwardExecution(i)
      val realGradOutput = m.element.asInstanceOf[MklDnnModule].
        initGradWPrimitives(findGradOutputFormats(m, lastGradInputFormats), phase)
//      lastGradInputFormats.zip(realGradOutput).foreach {
//        case (gi, go2) => reorderManager.register(gi, go2)
//      }
      if (i == 0) firstRealGradOutputFormats = realGradOutput
      lastGradInputFormats = m.element.asInstanceOf[MklDnnModule].gradInputFormats()
    }
    firstRealGradOutputFormats
  }
}