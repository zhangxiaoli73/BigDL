/*
 * Copyright 2018 Analytics Zoo Authors.
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


import com.intel.analytics.bigdl.nn.{CMulTable, CSubTable}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class InternalCSubTable[T: ClassTag](expandDim: Int = 1)
                                   (implicit ev: TensorNumeric[T]) extends CSubTable[T] {
  private var expandLayer: AbstractModule[Tensor[T], Tensor[T], T] = null

  override def updateOutput(input: Table): Tensor[_] = {
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)

    expandLayer = InternalExpand(input1.size())
    val input3 = expandLayer.forward(input2)

    output = super.updateOutput(T(input1, input3))
    return output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[_]): Table = {
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)

    expandLayer = InternalExpand(input1.size())
    val input3 = expandLayer.forward(input2)

    gradInput = super.updateGradInput(T(input1, input3), gradOutput)
    gradInput(2) = expandLayer.backward(input2, gradInput[Tensor[T]](2))
    gradInput
  }

  override def toString: String = s"InternalCSubTable()"

}

object InternalCSubTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : InternalCSubTable[T] = {
    new InternalCSubTable[T]()
  }
}
