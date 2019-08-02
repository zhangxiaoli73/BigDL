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

import org.scalatest.{FlatSpec, Matchers}

class FPN2MLPFeatureExtractorSpec extends FlatSpec with Matchers {
  "FPN2MLPFeatureExtractor" should "be ok" in {
    val resolution = 7
    val scales = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f)
    val sampling_ratio = 2.0f
    val representation_size = 1024
    val use_gen = false

    val layer = new FPN2MLPFeatureExtractor(resolution, scales,
      sampling_ratio, representation_size)


  }
}
