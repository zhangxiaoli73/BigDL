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

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

/**
 * The mean squared error criterion
 * e.g. input: a, target: b, total elements: n
 * loss(a, b) = 1/n \sum |a_i - b_i|^2
 * sizeAverage is true by default to divide the sum of squared error by n
 */
@SerialVersionUID(- 7078521754128606735L)
class MSECriterion[@specialized(Float, Double) T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  var sizeAverage = true

  @transient
  private var ones: Array[T] = null

  @transient
  private var buffer: Array[T] = null

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    output = ev.fromType[Int](0)

    gradInput.resizeAs(input)

    val nElement = input.nElement

    if (ones == null || ones.length < nElement) {
      ones = Array.fill(nElement)(ev.fromType[Int](1))
    }
    if (buffer == null || buffer.length < nElement) {
      buffer = new Array[T](nElement)
    }


    ev.vSub(nElement,
      input.storage.array,
      input.storageOffset - 1,
      target.storage.array,
      target.storageOffset - 1,
      gradInput.storage.array,
      gradInput.storageOffset - 1)

    ev.vPowx(nElement,
      gradInput.storage.array,
      gradInput.storageOffset - 1,
      ev.fromType[Int](2),
      buffer,
      0)

    output = ev.dot(nElement,
      ones,
      0,
      1,
      buffer,
      0,
      1)

    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](nElement))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val nElement = input.nElement
    var norm = ev.fromType[Int](2)
    if (sizeAverage) {
      norm = ev.fromType[Double](2.0 / nElement)
    }

    ev.scal(nElement,
      norm,
      gradInput.storage.array,
      gradInput.storageOffset - 1,
      1)

    gradInput
  }

}

object MSECriterion {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : MSECriterion[T] = {
    new MSECriterion[T]()
  }
}
