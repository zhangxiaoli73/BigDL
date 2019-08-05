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

import org.dmg.pmml.False
import org.scalatest.{FlatSpec, Matchers}

class MaskRCNNFPNFeatureExtractorSpec extends FlatSpec with Matchers {
  "MaskRCNNFPNFeatureExtractor" should "be ok" in {
    val resolution = 14
    val scales = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f)
    val sampling_ratio = 2
    val in_channels = 6
    val use_gn = false
    val layers = Array[Int](256, 256, 256, 256)
    val dilation = 1

    val layer = new MaskRCNNFPNFeatureExtractor(in_channels, resolution, scales,
      sampling_ratio, layers, dilation, use_gn)


  }
}
