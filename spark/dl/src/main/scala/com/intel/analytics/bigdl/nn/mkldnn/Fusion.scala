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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Node

private[mkldnn] object Fusion {

  private val fuse = true // System.getProperty("bigdl.mkldnn.fusion", "false").toBoolean

  def fuseModule(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (!fuse) return;
    node.element match {
      case relu: ReLU => fusionRelu(node)
      case bn: SpatialBatchNormalization => fusionBn(node)
      // case cadd: CAddTable => fusionCAddTable(node)
      case _ =>
    }
  }

  def fuseCAdd(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (!fuse) return;
    node.element match {
      case cadd: CAddTable => fusionCAddTable(node)
      case _ =>
    }
  }

  /**
    * fuse conv(without relu or bn fusion) with bn
    * if bn has fused with relu, then fuse relu and bn with conv
    * @param node
    */
  private def fusionBn(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    val bn = node.element.asInstanceOf[SpatialBatchNormalization]
    node.prevNodes.foreach(n => {
      n.element match {
        case conv : SpatialConvolution =>
          if (!conv.relu && !conv.batchNorm) {
            if (bn.relu) conv.setReLU(true)
            fusionConvBn(conv, bn)
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case _ => null
      }})
  }

  /**
    * fuse relu with conv or bn
    * @param node
    */
  private def fusionRelu(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    // if (node.element.asInstanceOf[ReLU].sum) return;
    node.prevNodes.foreach(n => {
      n.element match {
        case conv: SpatialConvolution =>
          if (!conv.relu) {
            conv.setReLU(true)
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case bn: SpatialBatchNormalization =>
          if (!bn.relu) {
            bn.setReLU(true)
            node.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]
          }
        case _ => null
      }})
  }

  private def findPrevious(node: Node[AbstractModule[Activity, Activity, Float]])
  : Node[AbstractModule[Activity, Activity, Float]] = {
    if (node.element.isInstanceOf[Identity] && node.prevNodes.length == 1) {
      findPrevious(node.prevNodes(0))
    } else node
  }

  /**
    * Fuse previous layers of CAddTable.
    * If one of previous layers is conv, and the other is relu or bn,
    * then fuse them together.
    * @param node
    */
  private def fusionCAddTable(node: Node[AbstractModule[Activity, Activity, Float]]): Unit = {
    if (node.element.isInstanceOf[CAddTable] && node.prevNodes.length == 2) {
      val previousNodes = node.prevNodes.toArray
      val node1 = findPrevious(previousNodes(0))
      val node2 = findPrevious(previousNodes(1))

      var conv : Node[Module[Float]] = null
      var other : Node[Module[Float]] = null
      var otherNumber: Int = 0

      if (node1.element.isInstanceOf[SpatialConvolution]) {
        if (!node1.element.asInstanceOf[SpatialConvolution].sum) conv = node1
        other = node2
        otherNumber = 1
      } else if (node2.element.isInstanceOf[SpatialConvolution]) {
        if (!node2.element.asInstanceOf[SpatialConvolution].sum) conv = node2
        other = node1
        otherNumber = 0
      }
      // meet fuse requirements
      if (conv != null && other != null) {
        node.element = conv.element
        node.element.asInstanceOf[SpatialConvolution].setSumOp(other.element, otherNumber)
        conv.element = Identity[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]]

        val nexts = node.nextNodes(0)
        if (nexts.element.isInstanceOf[ReLU]) {
          node.element.asInstanceOf[SpatialConvolution].setReLU(true)
          nexts.element = new Identity()
        }
      }
    }
  }

  private def fusionConvBn(conv: SpatialConvolution,
                           bn: SpatialBatchNormalization): Unit = {

    conv.setBatchNorm(true)
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
}
