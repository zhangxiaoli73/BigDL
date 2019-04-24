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

import com.intel.analytics.bigdl.nn.LookupTable
import com.intel.analytics.bigdl.nn.ops.Gather
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class EmbeddingSharedWeightsSpec extends FlatSpec with Matchers {
  val outputExpected = Tensor[Float](
    T(T(T(-1.1599494,   0.760163,   -0.40056285, -1.1687254,  -0.9138438,
    -0.36700436,  1.823892,   -1.1820021,   0.5161278,  -0.9247919),
    T(-1.3161435,   0.3335442,  -1.717504,    0.1719768,  -1.0396556,
    1.1057091,  -1.1911561,   1.0511508,   0.182776,    1.0882055),
    T( 0.4910144,  -0.15613313,  1.4748657,   0.43576995,  2.1953826,
      -0.835677,   -1.2184695,   0.80400825,  1.1463742,  -1.0150346),
    T( 0.3996748,   0.9506703,   1.1250867,  -0.8667602,  -1.1034116,
    2.3314137,   1.1097203,   0.71407086,  1.7064031,   1.8066634),
    T(-0.85047543, -2.599748,    1.2207254,  -2.0881205,  -0.19363593,
      -1.276643,   -0.02703438,  1.0847754,  -0.65506506,  0.4604092),
    T(-0.05895612,  0.26545224,  0.2933284,   0.42852548,  0.7240954,
    0.1287913,   1.4303886,   0.6864762,   2.1965477,   0.51878077),
    T( 0.71048903, -1.5525047,  -1.3863628,   0.3238582,   0.81226677,
    0.19209047, -0.23002781, -0.6363123,   1.0210168,   0.65428704),
    T(-0.28552356,  0.5651177,  -0.79568213,  0.07561763, -1.0208569,
    1.0577098,  -1.2571571,   0.42046708, -2.5739086,   0.9694715),
    T( 0.9546018,   1.5931981,  -0.44433123, -0.33717924,  0.7956628,
    0.50112695, -0.22244534,  1.7690458,  -0.898172,    1.8298535),
    T( 0.0090594,   0.01880967,  1.7577182,  -0.6437277,   0.3668765,
    0.15854552, -0.6759979,   0.53726906, -1.2015843,  -1.7854905),
    T(-0.7219142,  -0.10999311,  1.5652132,  -0.3543373,  -1.1179914,
    0.34499285,  0.6499864,  -1.6437156,   0.9259236,  -0.476595),
    T(-0.03769343,  0.52877223, -0.2173391,   1.1371078,  -0.59003806,
    1.5462487,  -0.6499737,  -1.0323933,   0.197083,    0.68658423)),
    T(T(-0.09307563, -0.55122334,  1.2353249,  -1.1112415,  -0.05812125,
    0.6815989,   0.691256,   -0.77127314, -0.10874277,  0.86469096),
    T(-0.7219142,  -0.10999311,  1.5652132,  -0.3543373,  -1.1179914,
    0.34499285,  0.6499864,  -1.6437156,   0.9259236,  -0.476595),
    T(-0.6199275,  -0.47378838,  0.8650373,   0.27147385,  0.3707318,
      -0.19951358,  0.7916733,  -0.33982825,  0.18631981, -1.5471507),
    T( 0.59465605, -0.39653215, -2.6116316,  -1.1501349,  -1.1990832,
    0.41783467, -0.22730026,  0.3142501,  -0.5856289,  -0.10131569),
    T(-0.13715318, -0.7494559,  -0.6900695,  -1.2961766,  -0.15865716,
    1.3895628,   0.90216327, -1.311854,   -0.15067385, -0.6309336),
    T(-0.03769343,  0.52877223, -0.2173391,   1.1371078,  -0.59003806,
    1.5462487,  -0.6499737,  -1.0323933,   0.197083,    0.68658423),
    T(-0.17206922,  1.0948895,   1.0302242,  -0.9556774,  -0.07595373,
      -1.4860637,   2.5717487,  -1.7536626,   1.1291388,   0.97053045),
    T( 1.0521581,   0.65624017, -1.290194,    0.64157075, -0.40509227,
      -0.65354455,  0.4234868,  -1.3410776,   0.05931387, -0.5433723),
    T( 0.17671813,  0.8072072,   1.3246931,   0.39417782, -0.23720965,
    0.9679637,  -1.0234876,  -0.8661555,  -1.5812052,  -0.37635),
    T(-1.1599494,   0.760163,   -0.40056285, -1.1687254,  -0.9138438,
      -0.36700436,  1.823892,   -1.1820021,   0.5161278,  -0.9247919),
    T( 0.3996748,   0.9506703,   1.1250867,  -0.8667602,  -1.1034116,
    2.3314137,   1.1097203,   0.71407086,  1.7064031,   1.8066634),
    T(-0.03769343,  0.52877223, -0.2173391,   1.1371078,  -0.59003806,
    1.5462487,  -0.6499737,  -1.0323933,   0.197083,    0.68658423))))


   "gather" should "correct" in {
     val vocab_size = 20
     val hidden = 10
     val batch = 2
     val length = 12
     val weight = Tensor[Float](vocab_size, hidden).fill(0.01f)
     val input = Tensor[Float](batch, length).fill(0.1f)

     val gather = new Gather[Float, Float]()

     val out = gather.forward(T(input, weight))

     println("done")
   }

  "look up table" should "correct" in {
    val arr = T(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 4, 12, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 7, 12)
    val input = Tensor[Float](arr).resize(Array(2, 12)).add(1.0f)

    val m = math.sqrt(10)
    val look = LookupTable[Float](20, 10)

    val out = look.forward(input)
    val gradInput = look.backward(input, out.contiguous())

    val tmp = 0
  }

  "EmbeddingSharedWeights" should "correct" in {
    val arr = T(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 4, 12, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 7, 12)
    val input = Tensor[Float](arr).resize(Array(2, 12)).add(1.0f)

    val look = new EmbeddingSharedWeights[Float](20, 10)

    val out = look.forward(input)
    out should be(outputExpected)
    val gradInput = look.backward(input, out.contiguous())

    println("done")

  }
}
