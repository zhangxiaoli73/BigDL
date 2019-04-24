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

import com.intel.analytics.bigdl.nn.Mean
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class LayerNormalizationSpec extends FlatSpec with Matchers {

  val input = Tensor[Float](
    T(T(1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763, -2.3015387,
    1.74481176, -0.7612069),
    T( 0.3190391,  -0.24937038,  1.46210794, -2.06014071, -0.3224172,  -0.38405435,
    1.13376944, -1.09989127))
  )

  val outputExpected = Tensor[Float](
      T(T( 0.2547953,  -0.00365025, -0.32806823,  0.32006884, -0.17416702, -0.92002285,
      -0.15028489, -0.56398183),
      T( 0.37940904, -0.04951846, -0.444961,    0.9273568,  -0.39561632, -0.28010735,
        -0.05768752, -0.73853105)))

  val gradInputExpected = Tensor[Float](
    T(T(-0.0655726, 0.11039984,  0.12039759,  0.00393196, -0.02003431, -0.09076728,
      0.00234376, -0.06069893),
    T(0.00566998, 0.14491531, -0.08142705, -0.09353723,  0.05779467,  0.03840649,
      -0.03802159, -0.03380056)))

  val gradWeightExpected = Tensor[Float](
    T( 0.5049854,   0.00593506, -0.5733794,  -1.8879533,  -0.06730913,  1.5727731,
      -0.28257257,  0.9264967))

  val gradBiasExpected = Tensor[Float](
    T( 0.6342044,  -0.05316871, -0.7730292,   1.2474256,  -0.56978333, -1.2001302,
      -0.2079724,  -1.3025129))

  "LayerNormalization layer" should "work correct" in {
    val layerNorm = new LayerNormalization[Float](8)

    val mean = Mean[Float](2, squeeze = false)
    val output = layerNorm.forward(input)
    val out2 = mean.forward(input)
    val grad2 = mean.backward(input, out2)
    // output should be(outputExpected)

    val gradInput = layerNorm.updateGradInput(input, output)

    println("done")
  }
}
