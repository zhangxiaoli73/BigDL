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

import breeze.linalg.{*, max}
import breeze.numerics.exp
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Sequential, _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

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
  def getPaddingBias[T: ClassTag](input: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val res = getPadding[T](input)
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

  private def range(length : Int): Array[Float] = {
    val arr = new Array[Float](length)
    var i = 0
    while (i < arr.length) {
      arr(i) = i.toFloat
      i += 1
    }
    arr
  }


  /**
   * Create an bias tensor to be added to attention logits.
   * Returns tensor with shape (1, 1, length, length)
   * @param length
   * @tparam T
   * @return
   */
  def attentionBiasLowerTriangle[T: ClassTag](length: Int)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val output = Tensor[T](length, length).zero()
    val tmp = Tensor[T](length).zero()
    var i = 1
    while (i < length) {
      tmp.zero()
      val arr = tmp.storage().array()
      var j = arr.length - 1
      while (j >= i) {
        arr(j) = ev.fromType(-1e9)
        j -= 1
      }
      val out = output.select(1, i).copy(tmp)
      i += 1
    }
    output.resize(Array(1, 1, length, length))
  }

  // Shift the second dimension of x right by one.
  def shiftRight3D[T: ClassTag](input: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    // todo: return shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    val output = Tensor[T]().resizeAs(input).zero()
    val index = input.size(2)
    output.narrow(2, 1, 1).zero()
    output.narrow(2, 2, index - 1).copy(input.narrow(2, 1, index - 1))
    output
  }

  /**
    * Args:
    * x: a Tensor with shape [batch, length, channels]
    * min_timescale: a float
    * max_timescale: a float
    * start_index: index of first position
    * Returns: a Tensor the same shape as x.
    * @param input
    * @param min_timescale
    * @param max_timescale
    * @tparam T
    * @return
    */
  def addTimingSignal1D[T: ClassTag](input: Tensor[T],
    min_timescale : Float = 1.0f,
    max_timescale: Float = 1.0e4f)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    // fisrt dim is batch
    val length = input.size(2)
    val channels = input.size(3)
    // get_timing_signal_1d, return (1, length, channels)
    val position = Tensor[Float](Storage(range(length)))
    val num_timescales = channels / 2
    val log_timescale_increment = math.log(max_timescale.toFloat / min_timescale.toFloat) /
      math.max(num_timescales.toFloat - 1, 1)
    // tf.range(num_timescales)
    val inv_timescales = new Array[Double](num_timescales)
    var i = 0
    while (i < inv_timescales.length) {
      inv_timescales(i) = min_timescale * math.exp(i * -log_timescale_increment).toDouble
      i += 1
    }
    val scaled_time = Tensor[Float](length, 2)

    scaled_time.select(2, 1).copy(position).mul(inv_timescales(0).toFloat)
    scaled_time.select(2, 2).copy(position).mul(inv_timescales(1).toFloat)

    val sinRes = scaled_time.clone().apply1(e => math.sin(e).toFloat)
    val cosRes = scaled_time.clone().apply1(e => math.cos(e).toFloat)

    val output = Tensor[Float](length, channels)

    output.narrow(2, 1, sinRes.size(2)).copy(sinRes)
    output.narrow(2, sinRes.size(2) + 1, cosRes.size(2)).copy(cosRes)

    val outTemp = output.asInstanceOf[Tensor[T]]
    val out = input.select(1, 1).add(outTemp)
    val out2 = input.select(1, 2).add(outTemp)
    return input
  }

}
