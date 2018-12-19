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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Node

import scala.reflect.ClassTag

private[mkldnn] class Fusion[T: ClassTag] {

  val fuseConvBn = System.getProperty("bigdl.mkldnn.fusion.convbn", "false").toBoolean
  val fuseBnRelu = System.getProperty("bigdl.mkldnn.fusion.bnrelu", "false").toBoolean
  val fuseConvRelu = System.getProperty("bigdl.mkldnn.fusion.convrelu", "false").toBoolean
  val fuseConvSum = System.getProperty("bigdl.mkldnn.fusion.convsum", "false").toBoolean

  private def fuseModule(node: Node[AbstractModule[Activity, Activity, T]]) = {
    // fuse relu & conv(false)/bn(false)
    if (node.element.isInstanceOf[ReLU]) {
      node.prevNodes.foreach(n => {
        n.element match {
          case conv
            if (conv.isInstanceOf[SpatialConvolution] && fuseConvRelu) =>
            val element = n.element.asInstanceOf[SpatialConvolution]
            if (!element.relu) {
              element.setReLU(true)
              node.element = Identity().asInstanceOf[AbstractModule[Activity, Activity, T]]
            }
          case bn
            if (bn.isInstanceOf[SpatialBatchNormalization] && fuseBnRelu) =>
            val element = n.element.asInstanceOf[SpatialBatchNormalization]
            if (!element.relu) {
              element.setReLU(true)
              node.element = Identity().asInstanceOf[AbstractModule[Activity, Activity, T]]
            }
          case _ => null
        }})
    }

    // fuse bn & conv(false)
    if (node.element.isInstanceOf[SpatialBatchNormalization]) {
      node.prevNodes.foreach(n => {
        n.element match {
          case s
            if (s.isInstanceOf[SpatialConvolution] && fuseConvBn) =>
            val element = n.element.asInstanceOf[SpatialConvolution]
            if (!element.relu && !element.batchNorm) {
              fusionConvBn(element, node.element.asInstanceOf[SpatialBatchNormalization])
              node.element = Identity().asInstanceOf[AbstractModule[Activity, Activity, T]]
            }
          case _ => null
        }})
    }
    // TODO: conv & CAddTable
    if (node.element.isInstanceOf[CAddTable]) {
      node.prevNodes.foreach(n => {

      })
    }
  }

  def fusionConvBn(conv: SpatialConvolution,
                           bn: SpatialBatchNormalization): Unit = {

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

  def fusinConvCaddTable(): Unit = {

  }
}
