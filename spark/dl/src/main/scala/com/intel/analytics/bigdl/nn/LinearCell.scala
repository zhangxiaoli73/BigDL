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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class LinearCell[T : ClassTag] (
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
  (implicit ev: TensorNumeric[T])
  extends Cell[T](Array(hiddenSize)) {

  val parallelTable = ParallelTable[T]()
  parallelTable.add(Linear(inputSize, hiddenSize))
  parallelTable.add(Linear(hiddenSize, hiddenSize))

  override var cell: AbstractModule[Activity, Activity, T] =
    Sequential[T]().add(parallelTable)

  override def toString(): String = {
    val str = "nn.LinearCell"
    str
  }

  /**
    * Clear cached activities to save storage space or network bandwidth. Note that we use
    * Tensor.set to keep some information like tensor share
    *
    * The subclass should override this method if it allocate some extra resource, and call the
    * super.clearState in the override method
    *
    * @return
    */
  override def clearState(): LinearCell.this.type = {
    super.clearState()
    cell.clearState()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[RnnCell[T]]

  override def equals(other: Any): Boolean = other match {
    case that: RnnCell[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        cell == that.cell
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), cell)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object LinearCell {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3)
  (implicit ev: TensorNumeric[T]) : LinearCell[T] = {
    new LinearCell[T](inputSize, hiddenSize)
  }
}