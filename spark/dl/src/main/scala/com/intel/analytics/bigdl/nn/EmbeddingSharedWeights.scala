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
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

/**
 * Get token embeddings of x.
 *  Args: x: An int64 tensor with shape [batch_size, length]
 *  Returns:
 *    embeddings: float32 tensor with shape [batch_size, length, embedding_size]
 *    padding: float32 tensor with shape [batch_size, length] indicating the
 *      locations of the padding tokens in x.
 * @param vocab_size
 * @param hidden_size
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
class EmbeddingSharedWeights[T: ClassTag](
  val vocab_size: Int, val hidden_size: Int)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  // todo: init with random normal
  // val weight = Tensor[T](vocab_size, hidden_size).rand()
  val weight = Tensor[T]( T(T( 0.0901417,  -0.25920567,  0.35886005, -0.79801846),
    T( 0.7101189,  -0.5279109,   0.24793072,  0.07292826),
    T(-0.00906177,  0.6962627,   0.37465635,  0.15718417),
    T(-0.11258064, -0.3311236,  -0.3180385,   0.58970255),
    T( 0.17320412, -0.49935055,  0.7124023,  -0.28340986),
    T(-0.33200186,  1.0381325,  -0.18797834,  0.5976197),
    T(-0.06744625, -0.23964763,  0.37403554, -0.4539435),
    T(-0.39824682,  0.18769431,  0.02896992, -0.7393345),
    T( 0.5590472,  -0.7522993,  -0.44121778, -0.1815617),
    T( 0.7071572,   0.27919358,  0.23945637, -0.17475012),
    T(-0.36576417, -1.6407981,  -0.5480189,   0.00637588),
    T( 0.3870772,   0.5724747,  -0.6339975,  -0.6118532),
    T(-0.08697614, -0.21675488, -0.13310283,  0.19130157),
    T( 0.06459922, -0.57852674,  0.9070809,   0.09887356),
    T( 0.8016945,  -0.09532502, -0.6059104,   0.74728966),
    T( 0.24903144,  0.06780083,  0.16405171, -0.29252014))
  )

  var gradWeight = Tensor[T](vocab_size, hidden_size).zero()

  private val value = ev.fromType(math.sqrt(hidden_size))
  private var maskFull : Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val inputBuffer = input.contiguous()
    output.index(1, inputBuffer.view(inputBuffer.nElement()), weight)
    output = output.view(inputBuffer.size(1), inputBuffer.size(2), weight.size(2))

    if (maskFull == null) maskFull = Tensor[T](input.size(1), input.size(2), weight.size(2))
    maskFull.select(3, 1).copy(inputBuffer).apply1(e => {if (e != ev.zero) ev.one else ev.zero})
    var i = 1
    // todo: optimize
    while (i <= hidden_size) {
      maskFull.select(3, i).copy(maskFull.select(3, 1))
      i += 1
    }
    output.cmul(maskFull)
    output.mul(value)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    //    gradInput.resizeAs(gradOutput).copy(gradOutput)
    //    gradInput.mul(value)
    //    gradInput.addcmul(ev.fromType[Int](1), maskFull, gradInput)

    // for look up table

    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }
    gradInput
  }

  // update weights
  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    val inputBuffer = input.contiguous()
    val _gradOutput = gradOutput.contiguous().mul(value)
    _gradOutput.addcmul(ev.fromType[Int](1), maskFull, _gradOutput)

    val input_data = inputBuffer.storage().array()
    val input_offset = inputBuffer.storageOffset() - 1
    val numEle = inputBuffer.nElement()

    if (scaleW != 0) {
      val gw = gradWeight.storage().array()
      val go = _gradOutput.storage().array()
      val stride = gradWeight.stride(1)

      var i = 0
      while (i < numEle) {
        val k = ev.toType[Int](input_data(i + input_offset)) - 1
        ev.axpy(stride, ev.fromType(scaleW), go, i * stride + _gradOutput.storageOffset() - 1, 1,
          gw, k * stride + gradWeight.storageOffset() - 1, 1)
        i += 1
      }
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }
}

object EmbeddingSharedWeights {
  def apply[T: ClassTag](vocab_size: Int, hidden_size: Int)
           (implicit ev: TensorNumeric[T]): EmbeddingSharedWeights[T] =
    new EmbeddingSharedWeights(vocab_size, hidden_size)
}