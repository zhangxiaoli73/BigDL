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
package com.intel.analytics.bigdl.nn.rnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.dmg.pmml.True

import scala.reflect.ClassTag

class LayerNormalization[T: ClassTag](hidden_size: Int)(implicit ev: TensorNumeric[T])
  extends BaseModule[T] {

  override var model : Module[T] = buildModel()

  private var weights = Tensor[T](hidden_size).fill(ev.one)
  private var bias = Tensor[T](hidden_size).fill(ev.zero)

  private def buildModel(): Module[T] = {
    val input = Input()
    val mean = Mean().inputs(input)
    val square = Square().inputs(input, mean)
    val variance = Mean().inputs(square)
    val graph = Graph(Array(input), Array(mean, variance))
    if (this.train) graph.training() else graph.evaluate()
    graph
  }

  override def updateOutput(input: Activity): Activity = {
    require(input.isTensor, "input should be tensor, but get table")
    val out = model.updateOutput(input)
    val mean = out.toTable.apply[Tensor[T]](1)
    val variance = out.toTable.apply[Tensor[T]](1)

    val out2 = reduceMeanForward(input.toTensor[T], mean, variance)

    out2.cmul(weights).add(bias)
    output
  }

  private def reduceMeanForward(inputX: Tensor[T],
                                inputY: Tensor[T], inputZ: Tensor[T]): Tensor[T] = {
    val epsion = ev.fromType(1e-6)
    inputZ.add(epsion)
    inputZ.sqrt()
    inputX.sub(inputY).cmul(inputZ)
  }
}
