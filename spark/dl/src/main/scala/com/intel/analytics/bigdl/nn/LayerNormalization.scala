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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.{Module => _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[nn] class LayerNormalization[T: ClassTag](hidden_size: Int)
  (implicit ev: TensorNumeric[T]) extends BaseModule[T] {
  override def buildModel(): Module[T] = {
    val input = Input()
    val mean = Mean(-1, squeeze = false).inputs(input)
    val sub = InternalCSubTable().inputs(input, mean)
    val square = Square().inputs(sub)
    val mean2 = Mean(-1, squeeze = false).inputs(square)
    val add = AddConstant(1e-6).inputs(mean2)
    val sqrt = Power(-0.5, 1, 0).inputs(add)
    val mul = InternalCMulTable().inputs(sub, sqrt)
    val linear = TimeDistributed(
      new LayerLinear[T](hidden_size)).inputs(mul)
    Graph(input, linear)
  }
  override def updateOutput(input: Activity): Activity = {
    output = model.updateOutput(input)
    output
  }
}

private[nn] class LayerLinear[T: ClassTag](hidden_size: Int)
   (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var weight = Tensor[T](hidden_size).fill(ev.one)
  var bias = Tensor[T](hidden_size).fill(ev.zero)
  var gradWeight = Tensor[T](hidden_size)
  var gradBias = Tensor[T](hidden_size)

  private val buffer = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    val size = output.size(1)
    var i = 1
    while(i <= size) {
      output.select(1, i).cmul(weight).add(bias)
      i += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    val size = output.size(1)
    var i = 1
    while(i <= size) {
      gradInput.select(1, i).addcmul(gradOutput.select(1, i), weight)
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    buffer.resizeAs(input).zero()
    buffer.addcmul(input, gradOutput)
    gradWeight = buffer.sum(1).squeeze()
    gradBias = gradOutput.sum(1).squeeze()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }
}
