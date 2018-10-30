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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, T}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class IRGraph[T: ClassTag](
    val inputs : Seq[Node[IRElement]],
    val outputs : Seq[Node[IRElement]],
    private[bigdl] val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] with Serializable {


  var input_layers = new ArrayBuffer[Node[IRElement]]
  var output_layers = new ArrayBuffer[Node[IRElement]]
  var layer_name_map = T()

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
    // build to generate BigDL graph
  }
}

object IRGraph {
//  def apply[T: ClassTag](
//    inputs: Seq[Node[TFElement]],
//    outputs: Seq[Node[TFElement]],
//    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None
//  )( implicit ev: TensorNumeric[T]): IRGraph[T] = {
//    new IRGraph[T](inputs.asInstanceOf[Seq[Node[IRElement]]],
//      outputs.asInstanceOf[Seq[Node[IRElement]]], variables)
//  }
  def apply[T: ClassTag](
    inputs: Seq[Node[IRElement]],
    outputs: Seq[Node[IRElement]],
    variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None
  )( implicit ev: TensorNumeric[T]): IRGraph[T] = {
    new IRGraph[T](inputs, outputs, variables)
  }
}