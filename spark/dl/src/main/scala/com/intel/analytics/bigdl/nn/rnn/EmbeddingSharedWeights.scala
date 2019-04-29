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

import breeze.math.Complex.scalar
import breeze.numerics.sqrt
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
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
class EmbeddingSharedWeights[T: ClassTag](vocab_size: Int, hidden_size: Int)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  val weight = Tensor[T](vocab_size, hidden_size).rand() // init with random normal

//  var weight = Tensor[T](
//    T(T( 0.33296704, -0.2825494,   0.09811679,  0.3889332,  -0.19899155,  0.1799927,
//      -0.5091036,   0.34838164, -0.25564954,  0.02378663),
//      T(-0.3668082,   0.24038465, -0.1266691,  -0.3695834,  -0.28898278, -0.11605697,
//        0.5767653,  -0.37378186,  0.16321395, -0.29244485),
//      T(-0.0294331,  -0.17431213,  0.390644,   -0.3514054,  -0.01837955,  0.2155405,
//        0.21859433, -0.24389797, -0.03438748,  0.2734393),
//      T(-0.4162011,   0.10547593, -0.5431225,   0.05438384, -0.32876796,  0.3496559,
//        -0.37667665,  0.33240306,  0.05779885,  0.34412077),
//      T(-0.2282893,  -0.03478288,  0.49496388, -0.11205129, -0.35353994,  0.10909632,
//        0.20554374, -0.5197885,   0.29280275, -0.15071258),
//      T( 0.15527238, -0.04937363,  0.46639347,  0.13780256,  0.6942409,  -0.2642643,
//        -0.3853139,   0.25424972,  0.36251536, -0.3209821),
//      T(-0.19603829, -0.14982504,  0.2735488,   0.08584757,  0.11723569, -0.06309173,
//        0.25034907, -0.10746313,  0.0589195,  -0.48925203),
//      T( 0.12638827,  0.30062833,  0.35578364, -0.27409363, -0.34892938,  0.7372578,
//        0.35092437,  0.22580902,  0.53961205,  0.57131714),
//      T( 0.18804675, -0.12539448, -0.82587045, -0.3637046,  -0.3791834,   0.13213092,
//        -0.07187865,  0.09937461, -0.18519212, -0.03203883),
//      T(-0.26894394, -0.82211244,  0.38602728, -0.66032165, -0.06123305, -0.40370998,
//        -0.00854902,  0.34303612, -0.20714976,  0.14559416),
//      T(-0.04337164, -0.23699877, -0.21821913, -0.40988702, -0.0501718,   0.43941835,
//        0.28528908, -0.41484466, -0.04764726, -0.19951871),
//      T(-0.01864356,  0.08394337,  0.09275859,  0.13551165,  0.22897907,  0.04072738,
//        0.4523286,   0.21708283,  0.6946094,   0.16405289),
//      T(-0.01191971,  0.16721246, -0.06872866,  0.35958508, -0.18658641,  0.48896676,
//        -0.20553973, -0.32647145,  0.06232312,  0.217117),
//      T( 0.22467636, -0.49094507, -0.4384064,   0.10241295,  0.2568613,   0.06074434,
//        -0.07274118, -0.20121962,  0.32287386,  0.20690373),
//      T(-0.05441307,  0.34623447,  0.3257855,  -0.30221173, -0.02401868, -0.4699346,
//        0.81325835, -0.5545568,   0.35706505,  0.30690867),
//      T(-0.09029048,  0.17870592, -0.25161678,  0.02391239, -0.3228233,   0.33447722,
//        -0.39754796,  0.13296336, -0.81394136,  0.3065738),
//      T( 0.33272162,  0.20752136, -0.40799516,  0.20288248, -0.12810142, -0.20666893,
//        0.13391829, -0.42408597,  0.01875669, -0.1718294),
//      T( 0.3018716,   0.50381345, -0.14050987, -0.10662544,  0.25161067,  0.15847026,
//        -0.07034339,  0.5594214,  -0.28402692,  0.5786505),
//      T( 0.05588318,  0.25526133,  0.41890472,  0.12464997, -0.07501227,  0.306097,
//        -0.3236552,  -0.27390242, -0.500021,   -0.11901231),
//      T( 0.00286483,  0.00594814,  0.5558393,  -0.20356458,  0.11601654,  0.0501365,
//        -0.2137693,   0.16989939, -0.37997434, -0.5646217)))

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
