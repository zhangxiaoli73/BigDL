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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * A Simple layer selecting an index of the input tensor in the given dimension
 *
 * @param dimension the dimension to select
 * @param index the index of the dimension to be selected
 */

@SerialVersionUID(1581502108010704056L)
class Select[T: ClassTag](
  dimension: Int,
  index: Int
)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  val buffer = Tensor[T]()

  def getPositiveDimAndIndex(input: Tensor[T]): (Int, Int) = {
    val dim = if (dimension < 0) {
      input.dim() + dimension + 1
    } else {
      dimension
    }

    val index = if (this.index < 0) {
      input.size(dim) + this.index + 1
    } else {
      this.index
    }
    (dim, index)
  }

  def copyMemoryOld(src: Tensor[T], dst: Tensor[T], srcIndex: Int): Unit = {
    val srcSize = src.size()
    val batchSize = srcSize(0)
    val timeSize = srcSize(1)
    val stepSize = src.nElement() / (batchSize * timeSize)
    val srcArr = src.storage().array()
    val srcOffset = src.storageOffset() - 1

//    srcSize(0) = timeSize
//    srcSize(1) = batchSize
//    dst.resize(srcSize)
    val dstArr = dst.storage().array()
    val dstOffset = dst.storageOffset() - 1

    var t = 1
    val l = (srcIndex-1) * stepSize
    while (t <= batchSize) {
      val length1 = timeSize * stepSize * (t-1) + srcOffset
      val length2 = (t-1) * stepSize + dstOffset
      System.arraycopy(srcArr, length1 + l, dstArr, l * batchSize + length2, stepSize)
      t += 1
    }
  }

  def copyMemory(src: Tensor[T], dst: Tensor[T], srcIndex: Int): Unit = {
    val batchSize = src.size(1)
    val timeSize = src.size(2)
    val stepSize = src.nElement() / (batchSize * timeSize)

    val srcArr = src.storage().array()
    var srcOffset = src.storageOffset() - 1
    val dstArr = dst.storage().array()
    var dstOffset = dst.storageOffset() - 1

    val recordSize = timeSize * stepSize
    val indexSize = (srcIndex-1) * stepSize

    var b = 0
    while (b < batchSize) {
      System.arraycopy(srcArr, srcOffset + indexSize, dstArr, dstOffset, stepSize)
      srcOffset += recordSize
      dstOffset += stepSize
      b += 1
    }
  }

  def copyToMemoryOld(src: Tensor[T], dst: Tensor[T], dstIndex: Int): Unit = {
    val dstArr = dst.storage().array()
    val dstOffset = dst.storageOffset() - 1
    val batchSize = dst.size(1)
    val times = dst.size(2)
    val stepSize = dst.nElement() / (batchSize * times)

    val batchStepSize = batchSize * stepSize
    val srcArr = src.storage().array()
    val srcOffset = src.storageOffset() - 1
    val length1 = (dstIndex - 1) * stepSize + dstOffset
    var l = 0
    while (l < batchStepSize) {
      System.arraycopy(srcArr, l + srcOffset, dstArr, times * l + length1, stepSize)
      l += stepSize
    }
  }

  def copyToMemory(src: Tensor[T], dst: Tensor[T], dstIndex: Int): Unit = {
    val batchSize = dst.size(1)
    val timeSize = dst.size(2)
    val stepSize = dst.nElement() / (batchSize * timeSize)

    val dstArr = dst.storage().array()
    var dstOffset = dst.storageOffset() - 1
    val srcArr = src.storage().array()
    var srcOffset = src.storageOffset() - 1

    val recordSize = timeSize * stepSize
    val indexSize = (dstIndex - 1) * stepSize

    var b = 0
    while (b < batchSize) {
      System.arraycopy(srcArr, srcOffset, dstArr, dstOffset + indexSize, stepSize)
      srcOffset += stepSize
      dstOffset += recordSize
      b += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val (dim, index) = getPositiveDimAndIndex(input)
    val output = input.select(dim, index)
    this.output.resizeAs(output)
    if ((dim == 2) && (input.dim() > 2)) {
      copyMemory(input, this.output, index)
    } else {
      this.output.copy(output)
    }
    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val (dim, index) = getPositiveDimAndIndex(input)
    gradInput.resizeAs(input)
    gradInput.zero()
    if ((dim == 2) && (gradInput.dim() > 2)) {
      copyToMemory(gradOutput, gradInput, index)
    } else {
      gradInput.select(dim, index).copy(gradOutput)
    }
    gradInput
  }

  override def toString: String = s"nn.Select"
}

object Select {
  def apply[@specialized(Float, Double) T: ClassTag](
      dimension: Int,
      index: Int)(implicit ev: TensorNumeric[T]) : Select[T] = {
    new Select[T](dimension, index)
  }
}
