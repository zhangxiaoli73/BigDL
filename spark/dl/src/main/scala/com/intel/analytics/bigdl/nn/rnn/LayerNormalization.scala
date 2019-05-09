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
import breeze.numerics.sqrt
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.tf.Const
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[nn] class LayerNormalization[T: ClassTag](hidden_size: Int, dims: Int = 3)
  (implicit ev: TensorNumeric[T]) extends BaseModule[T] {
  override def buildModel(): Module[T] = {
    val input = Input()
    // val mean = Mean(3, squeeze = false).inputs(input)
    val mean = TimeDistributed(Mean(2, squeeze = false)).inputs(input)
    val sub = InternalCSubTable().inputs(input, mean)
    val square = Square().inputs(sub)
    // val mean2 = Mean(3, squeeze = false).inputs(square)
    val mean2 = TimeDistributed(Mean(2, squeeze = false)).inputs(square)
    val add = AddConstant(1e-6).inputs(mean2)
    val sqrt = Power(-0.5, 1, 0).inputs(add)
    val mul = InternalCMulTable().inputs(sub, sqrt)
    val linear = TimeDistributed(
      new LayerLinear[T](hidden_size, dims).setName(this.getName())).inputs(mul)
    Graph(input, linear)
  }
}
