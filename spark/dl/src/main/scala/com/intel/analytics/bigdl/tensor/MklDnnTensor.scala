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

package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class MklDnnTensor[T: ClassTag](
  private[tensor] var _size: Array[Int],
  private[tensor] var _stride: Array[Int],
  var nDimension: Int)(implicit ev: TensorNumeric[T])
  extends QuantizedTensorUnsupported[T] {

  @transient var _pointer = allocate(size().product)

  val CACHE_LINE_SIZE = 64

  def allocate(capacity: Int): Long = {
    MklDnn.MemoryAlignedMalloc(capacity, CACHE_LINE_SIZE)
  }

  def release(): Unit = {
    MklDnn.MemoryAlignedFree(_pointer)
  }

  override def dim(): Int = nDimension

  override def size(): Array[Int] = _size

  override def size(dim: Int): Int = _size(dim)

  override def stride(): Array[Int] = _stride

  override def stride(dim: Int): Int = _stride(dim)

  override def nElement(): Int = _size.product

  override def set(other: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException
  }

  override def set(): Tensor[T] = {
    throw new UnsupportedOperationException
    this
  }

  override def copy(other: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException
    this
  }

  override def getTensorNumeric(): TensorNumeric[T] = ev

  override def getTensorType: TensorType = MklDnnType
}
