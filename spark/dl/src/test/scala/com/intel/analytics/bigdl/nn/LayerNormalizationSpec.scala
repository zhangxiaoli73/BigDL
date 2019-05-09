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

import com.intel.analytics.bigdl.nn.mkldnn.Equivalent
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
  val weightsExpected = Tensor[Float](
    T(-0.14037117, -0.16902402, -0.06451887, -0.5642037,
    0.24212438, 0.44951588, -0.4296978, 0.423163))

  val biasExpected = Tensor[Float](
    T(0.44111532, -0.06523705, -0.3474969, -0.08237404,
      -0.3565278, -0.18157673, 0.4592312, -0.36194998))

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
      -0.28257257,  0.9264967, 0.6342044,  -0.05316871, -0.7730292,   1.2474256,
      -0.56978333, -1.2001302, -0.2079724,  -1.3025129))

  "LayerNormalization layer" should "work correct" in {
    val layerNorm = new LayerNormalization[Float](8)
    val params = layerNorm.parameters()
    params._1.apply(0).copy(weightsExpected)
    params._1.apply(1).copy(biasExpected)

    val output = layerNorm.forward(input)
    output should be(outputExpected)

    val gradInput = layerNorm.backward(input, output)
    Equivalent.nearequals(gradInput.toTensor[Float], gradInputExpected)

    val gradWeights = layerNorm.getParameters()._2

    gradWeights should be(gradWeightExpected)
  }

  "LayerNormalization layer for 3 dims" should "work correct" in {
    val layerNorm = new LayerNormalization[Float](8)
    val params = layerNorm.parameters()
    params._1.apply(0).copy(weightsExpected)
    params._1.apply(1).copy(biasExpected)

    val input = Tensor[Float](T(T(
      T( 1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763,
      -2.3015387,   1.74481176, -0.7612069),
      T( 0.3190391,  -0.24937038,  1.46210794, -2.06014071, -0.3224172,
        -0.38405435,  1.13376944, -1.09989127),
      T(-0.17242821, -0.87785842,  0.04221375,  0.58281521, -1.10061918,
        1.14472371,  0.90159072,  0.50249434)),
      T(T( 0.90085595, -0.68372786, -0.12289023, -0.93576943, -0.26788808,
      0.53035547, -0.69166075, -0.39675353),
      T(-0.6871727,  -0.84520564, -0.67124613, -0.0126646,  -1.11731035,
      0.2344157,   1.65980218,  0.74204416),
      T(-0.19183555, -0.88762896, -0.74715829,  1.6924546,   0.05080775,
      -0.63699565,  0.19091548,  2.10025514))))

    val output = layerNorm.forward(input)
    // output should be(outputExpected)

    val gradInput = layerNorm.backward(input, output)
    // Equivalent.nearequals(gradInput.toTensor[Float], gradInputExpected)

    val gradWeights = layerNorm.getParameters()._2

    // gradWeights should be(gradWeightExpected)
  }

  "layer linear" should "work correct" in {
    val weight = Tensor[Float](T(-0.14037117, -0.16902402, -0.06451887,
      -0.5642037, 0.24212438, 0.44951588, -0.4296978, 0.423163))
    val bias = Tensor[Float](T(0.44111532, -0.06523705, -0.3474969,
      -0.08237404, -0.3565278, -0.18157673, 0.4592312, -0.36194998))

    val layer = new LayerLinear[Float](8, 2)
    layer.setWeightsBias(Array(weight, bias))
    val output = layer.forward(input)
    val outputExpected = Tensor[Float](
        T(T( 0.21310404,  0.03816448, -0.31341985,  0.5229988,  -0.14699152, -1.2161549,
          -0.2905106,  -0.6840646),
        T( 0.39633143, -0.02308746, -0.44183046,  1.0799649,  -0.43459287, -0.35421526,
          -0.02794704, -0.8273833)))

    output should be(outputExpected)

    val gradInput = layer.updateGradInput(input, output)
    val gradInputExpected = Tensor[Float](
      T(T(-0.02991366, -0.00645071,  0.02022149, -0.29507786, -0.03559023, -0.5466809,
      0.12483177, -0.28947085),
      T(-0.05563351,  0.00390234,  0.0285064,  -0.60932016, -0.10522553, -0.15922539,
      0.01200878, -0.35011798)))

    gradInput should be(gradInputExpected)

    layer.accGradParameters(input, output)

    val gradWeightExpected = Tensor[Float](T(0.4725998,  -0.01759003, -0.48046428,
      -2.7860408,   0.01291263,  2.9350655, -0.5385718,   1.4307464, 0.60943544,
      0.01507702, -0.75525033,  1.6029637,  -0.5815844,  -1.5703702, -0.31845763, -1.5114479))

    val gradWeights = layer.getParameters()._2

    gradWeights should be(gradWeightExpected)

    println("done")
  }

  "InternalMulTable" should "good" in {

    val input1 = Tensor[Float](T(T(-0.52817175, -1.07296862,  0.86540763, -2.3015387,
      1.74481176, -0.7612069, 0.3190391,  -0.24937038),
    T( 1.46210794, -2.06014071, -0.3224172,  -0.38405435,  1.13376944, -1.09989127,
      -0.17242821, -0.87785842)))
    val input2 = Tensor[Float](T(T(1.62434536), T(-0.61175641)))
    val layer = InternalCMulTable[Float]()
    val output = layer.forward(T(input1, input2))
    println(output)
    val gradInput = layer.backward(T(input1, input2), output)
    println(gradInput)
    print("done")
  }
}
