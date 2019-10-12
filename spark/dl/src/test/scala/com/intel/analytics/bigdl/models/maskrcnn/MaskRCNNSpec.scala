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

package com.intel.analytics.bigdl.models.maskrcnn

import com.intel.analytics.bigdl.nn.Nms
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}
import org.scalatest.{FlatSpec, Matchers}

class MaskRCNNSpec extends FlatSpec with Matchers {
  "build maskrcnn" should "be ok" in {
    RandomGenerator.RNG.setSeed(100)
    val resNetOutChannels = 32
    val backboneOutChannels = 32
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels)
    mask.evaluate()
    val input = Tensor[Float](1, 3, 224, 256).rand()
    val output = mask.forward(T(input, Tensor[Int](T(T(224, 256)))))
  }

  "NMS" should "be ok" in {
    val boxes = Tensor[Float](T(
      T(18.0357, 0.0000, 41.2893, 37.1173),
      T(30.0285, 6.2588, 53.1850, 39.0000),
      T(26.0422, 0.0000, 49.1954, 39.0000),
      T( 5.9485, 14.0573, 29.1708, 39.0000),
      T(42.0456, 0.0000, 57.0000, 37.1553),
      T(21.9588, 14.0357, 45.1161, 39.0000),
      T( 6.0533, 0.0000, 29.4083, 39.0000),
      T( 2.0541, 2.3791, 25.4243, 39.0000),
      T(14.0495, 2.3053, 37.3108, 39.0000),
      T(46.0309, 6.4025, 57.0000, 39.0000),
      T(22.0302, 2.4089, 45.1933, 39.0000),
      T(13.9671, 14.0175, 37.1495, 39.0000),
      T(10.0404, 0.0000, 33.3284, 33.2829),
      T(34.0374, 0.0000, 57.0000, 36.9072),
      T(38.0379, 6.2769, 57.0000, 39.0000),
      T(41.9751, 14.0583, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 13.2693, 33.3124),
      T(38.0422, 0.0000, 57.0000, 28.9761),
      T( 0.0000, 14.0690, 17.1186, 39.0000),
      T( 0.0000, 6.0356, 13.2223, 39.0000),
      T( 0.0000, 0.0000, 17.3122, 39.0000),
      T(22.0270, 0.0000, 45.1928, 25.2032),
      T(46.0094, 0.0000, 57.0000, 33.0826),
      T( 0.0000, 0.0000, 33.7101, 13.0355),
      T( 2.0302, 0.0000, 25.4260, 25.4481),
      T(42.0226, 0.0000, 57.0000, 25.1449),
      T(30.0364, 0.0000, 53.0853, 25.0766),
      T(14.0171, 0.0000, 37.2881, 25.2999),
      T(34.0521, 0.0000, 57.0000, 12.9051),
      T( 0.0000, 3.8999, 57.0000, 39.0000),
      T( 2.0133, 0.0000, 49.6427, 12.9898),
      T(28.0456, 0.0000, 57.0000, 39.0000),
      T( 0.0000, 11.8925, 47.3868, 39.0000),
      T( 8.0708, 11.9606, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 27.2810, 39.0000),
      T( 0.0000, 0.0000, 47.4577, 35.2592),
      T( 0.0000, 0.0000, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 57.0000, 39.0000),
      T(21.9457, 0.0000, 57.0000, 12.8811),
      T( 0.0000, 0.0000, 57.0000, 39.0000),
      T( 0.0000, 0.0000, 57.0000, 27.0690),
      T(13.8674, 22.0563, 44.9398, 39.0000),
      T(33.8700, 25.9730, 57.0000, 39.0000),
      T( 0.0000, 22.0516, 20.9330, 39.0000),
      T(41.9213, 21.9873, 57.0000, 39.0000),
      T(17.8165, 0.0000, 57.0000, 16.8779),
      T( 1.7646, 18.1004, 32.9480, 39.0000),
      T(11.8512, 0.0000, 57.0000, 35.4317),
      T(29.8503, 22.0435, 57.0000, 39.0000),
      T( 9.7594, 18.0566, 40.9166, 39.0000),
      T(33.7746, 1.9632, 57.0000, 24.9071),
      T( 0.0000, 14.0776, 24.9558, 39.0000),
      T(21.7241, 18.0735, 52.8998, 39.0000),
      T( 0.0000, 0.0000, 29.2906, 29.5339),
      T(41.8249, 0.0000, 57.0000, 17.0812),
      T( 0.0000, 0.0000, 17.3257, 17.4717),
      T( 0.0000, 0.0000, 17.1572, 25.5946),
      T( 0.0000, 0.0000, 45.4454, 17.0065),
      T( 0.0000, 2.0042, 21.2122, 33.4895),
      T(37.8946, 18.1178, 57.0000, 39.0000),
      T( 0.0000, 5.9850, 25.1862, 29.1060),
      T( 1.7353, 6.0499, 33.1671, 37.4231),
      T(21.6518, 26.0054, 57.0000, 39.0000),
      T( 5.7049, 0.0000, 37.2819, 29.4436),
      T(29.7011, 14.0272, 57.0000, 39.0000),
      T(17.7255, 0.0000, 49.0772, 29.2946),
      T(29.6133, 9.9153, 57.0000, 32.7949),
      T( 0.0000, 26.0193, 32.8463, 39.0000),
      T(17.6348, 10.0788, 48.9423, 39.0000),
      T(21.6906, 2.1241, 52.9483, 33.3707),
      T( 5.6194, 0.0000, 53.3307, 21.0163),
      T(13.8104, 0.0000, 45.2210, 17.3200),
      T(13.5956, 9.9687, 57.0000, 32.8566),
      T( 5.7003, 10.0389, 37.0897, 39.0000),
      T(13.7149, 2.0202, 45.0843, 33.2768),
      T( 9.7322, 5.9888, 41.1038, 37.3045),
      T( 5.5910, 26.0368, 52.8697, 39.0000),
      T(29.7840, 0.0000, 57.0000, 17.1027),
      T( 5.7736, 0.0000, 37.3917, 17.4214),
      T( 0.0000, 13.9622, 36.9701, 36.8555),
      T( 0.0000, 9.9967, 45.0663, 32.9533),
      T( 0.0000, 0.0000, 33.2938, 21.2008),
      T( 0.0000, 0.0000, 25.3888, 17.4817),
      T(21.7062, 0.0000, 53.0319, 21.2508),
      T( 9.6736, 0.0000, 41.2481, 21.3898),
      T( 0.0000, 1.9933, 37.2186, 25.1230),
      T( 5.5202, 5.9523, 53.1432, 28.9392),
      T(25.5138, 5.9795, 57.0000, 28.8653),
      T( 0.0000, 10.0011, 28.9181, 33.0324),
      T( 5.5488, 14.0092, 52.8771, 36.8956),
      T( 9.5096, 1.9473, 57.0000, 24.9822),
      T(17.5084, 13.9728, 57.0000, 36.8385),
      T( 0.0000, 22.0156, 40.7790, 39.0000),
      T(17.5165, 22.0209, 57.0000, 39.0000),
      T( 9.5040, 17.9792, 56.7784, 39.0000),
      T( 0.0000, 5.9792, 41.1165, 29.0066)))

    val scores = Tensor[Float](
      T(0.1117, 0.8158, 0.2626, 0.4839, 0.6765, 0.7539, 0.2627, 0.0428, 0.2080,
      0.1180, 0.1217, 0.7356, 0.7118, 0.7876, 0.4183, 0.9014, 0.9969, 0.7565,
      0.2239, 0.3023, 0.1784, 0.8238, 0.5557, 0.9770, 0.4440, 0.9478, 0.7445,
      0.4892, 0.2426, 0.7003, 0.5277, 0.2472, 0.7909, 0.4235, 0.0169, 0.2209,
      0.9535, 0.7064, 0.1629, 0.8902, 0.5163, 0.0359, 0.6476, 0.3430, 0.3182,
      0.5261, 0.0447, 0.5123, 0.9051, 0.5989, 0.4450, 0.7278, 0.4563, 0.3389,
      0.6211, 0.5530, 0.6896, 0.3687, 0.9053, 0.8356, 0.3039, 0.6726, 0.5740,
      0.9233, 0.9178, 0.7590, 0.7775, 0.6179, 0.3379, 0.2170, 0.9454, 0.7116,
      0.1157, 0.6574, 0.3451, 0.0453, 0.9798, 0.5548, 0.6868, 0.4920, 0.0748,
      0.9605, 0.3271, 0.0103, 0.9516, 0.2855, 0.2324, 0.9141, 0.7668, 0.1659,
      0.4393, 0.2243, 0.8935, 0.0497, 0.1780, 0.3011))

    val thresh = 0.5f
    val inds = new Array[Int](scores.nElement())
    val nms = new Nms
    val keepN = nms.nms(scores, boxes, thresh, inds)

    val expectedOutput = Array[Float](2.0f, 5.0f, 8.0f, 9.0f, 16.0f,
      21.0f, 23.0f, 24.0f, 25.0f, 36.0f, 42.0f, 43.0f, 49.0f, 55.0f,
      64.0f, 76.0f, 77.0f, 84.0f, 87.0f, 88.0f)

    for (i <- 0 to keepN - 1) {
      require(expectedOutput.contains(inds(i) - 1), s"${i} ${inds(i)}")
    }
  }
}

class MaskRCNNSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val resNetOutChannels = 32
    val backboneOutChannels = 32
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels).setName("MaskRCNN")
    mask.evaluate()
    val input = Tensor[Float](1, 3, 224, 256).rand()
    val output = mask.forward(input)

    runSerializationTest(mask, input)
  }
}
