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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Sequential, _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[rnn] object TransformerOperation {
  def dense[T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    bias: Boolean = true,
    activation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    name: String = "")(implicit ev: TensorNumeric[T]): Module[T] = {
    val seq = new Sequential[T]()
    val layer = Linear[T](
      inputSize = inputSize,
      outputSize = outputSize,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)

    layer.setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
    if (name != "") layer.setName(name)
    seq.add(TimeDistributed[T](layer))
    if (activation != null) seq.add(activation)
    seq
  }

  def softMax[T: ClassTag]()(implicit ev: TensorNumeric[T]): Module[T] = {
    val layer = SoftMax[T]()
    val model = Sequential[T]()
    model.add(Transpose[T](Array((2, 4))))
    model.add(layer)
    model.add(Transpose[T](Array((2, 4))))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  /**
    * Calculate bias tensor from padding values in tensor.
    * Bias tensor that is added to the pre-softmax multi-headed attention logits,
    * which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    * non-padding locations, and -1e9 (negative infinity) at padding locations.
    * Args: x: int tensor with shape [batch_size, length]
    * Returns: Attention bias tensor of shape [batch_size, 1, 1, length].
    * @param input
    * @tparam T
    * @return
    */
  def getPaddingBias[T: ClassTag](input: Tensor[T]): Tensor[T] = {
    val res = getPadding(input)
    res.addSingletonDimension(res, 2)
    res.addSingletonDimension(res, 3)
  }

  /**
   * Return float tensor representing the padding values in x.
   * Args:
   * x: int tensor with any shape
   * padding_value: int value that
   * Returns:float tensor with same shape as x containing values 0 or 1.
   *   0 -> non-padding, 1 -> padding
   */
  def getPadding[T: ClassTag](input: Tensor[T], paddingValue: Float = 0.0f)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    input.apply1(e => {if (e == paddingValue) ev.one else ev.zero})
  }

  /**
    * Return positional encoding.
    * Calculates the position encoding as a mix of sine and cosine functions with
    * geometrically increasing wavelengths.
    * Defined and formulized in Attention is All You Need, section 3.5.
    * Args:
    *  length: Sequence length.
    *  hidden_size: Size of the
    *  min_timescale: Minimum scale that will be applied at each position
    *  max_timescale: Maximum scale that will be applied at each position
    * Returns:
    *  Tensor with shape [length, hidden_size]
    * @param length
    * @param hidden_size
    * @param min_timescale
    * @param max_timescale
    * @tparam T
    */
//  def getPositionEncoding[T: ClassTag](
//     length: Int,
//     hidden_size: Int,
//     min_timescale: Float = 1.0f,
//     max_timescale: Float = 1.0e4f): Unit = {
//    val position = range(length)
//    val  num_timescales = hidden_size // 2
//    val log_timescale_increment = (math.log(max_timescale / min_timescale) / (num_timescales - 1))
//    val inv_timescales = min_timescale * math.exp(range(num_timescales) * -log_timescale_increment)
//    val scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
//    val signal = tf.concat([math.sin(scaled_time), math.cos(scaled_time)], axis=1)
//    return signal
//  }

  private def range(length : Int): Array[Int] = {
    val arr = new Array[Int](length)
    var i = 0
    while (i < arr.length) {
      arr(i) = i
      i += 1
    }
    arr
  }




}
