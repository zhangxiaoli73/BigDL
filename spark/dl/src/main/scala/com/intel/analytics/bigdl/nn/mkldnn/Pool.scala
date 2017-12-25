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

import breeze.linalg.{*, product}
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, SpatialMaxPooling, VariableFormat}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.sql.catalyst.expressions.Conv
import spire.std.float

import scala.reflect.ClassTag

class Pooling[T: ClassTag](
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0)(implicit ev: TensorNumeric[T]) extends TensorModule[Float] with Initializable {

  val convEngine = MklDnnOps.engineCreate(0)

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    gradInput
  }
}

object Pooling {
  def apply[@specialized(Float, Double) T: ClassTag](
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0)
    (implicit ev: TensorNumeric[T]): Pooling[T] = {
    new Pooling[T](kW, kH, dW, dH, padW, padH)
  }
}
