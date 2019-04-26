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

import breeze.macros.expand
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class LayerNormalization[T: ClassTag](hidden_size: Int)(implicit ev: TensorNumeric[T])
  extends BaseModule[T] {

  private val epsilon = 1e-6

  override def buildModel(): Module[T] = {
    val input = Input()
    val mean = Mean(2, squeeze = false).inputs(input)
    // val expand = new Expand(2, 8).inputs(mean)
    // val sub = CSubTable().inputs(input, expand)
    val sub = CSubTable().inputs(input, mean)
    val square = Square().inputs(sub)
    val mean2 = Mean(2, squeeze = false).inputs(square)
    val add = AddConstant(epsilon).inputs(mean2)
    val sqrt = Sqrt().inputs(add)
    val linear = new LayerLinear[T](hidden_size).inputs(sqrt)
    Graph(input, sub)
  }
}
