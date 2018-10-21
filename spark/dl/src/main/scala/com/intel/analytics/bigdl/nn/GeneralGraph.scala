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


import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.mkldnn.NodeType
import com.intel.analytics.bigdl.nn.mkldnn.NodeType.BigDLNode
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._

import scala.reflect.ClassTag

import java.util.List

class GeneralGraph[T: ClassTag, D](
   listNodes: List[Node[D]],
   _inputs: Seq[String] = null,
   _outputs: Seq[String] = null,
   node: NodeType = BigDLNode)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] with Serializable {

  def inputs(): Seq[String] = _inputs
  def outputs(): Seq[String] = _outputs

  def nodeType(): NodeType = nodeType
  def nodes(): List[Node[D]] = listNodes

  private var graph: Graph[T] = null

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
}
