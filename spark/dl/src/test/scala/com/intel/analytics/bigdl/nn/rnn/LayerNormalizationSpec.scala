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
    T(T( 1.3273382,  -0.3643671,  -0.3011318,  -0.7132936,   0.75316983, -1.6427587,
    1.4184761,  -0.47743273),
    T( 0.4395937,  -0.09299618,  1.5106294,  -1.7896566,  -0.16143979, -0.21919276,
    1.202982,   -0.88991964))
  )

  "LayerNormalization layer" should "work correct" in {
    val layerNorm = new LayerNormalization[Float](8)

    val output = layerNorm.forward(input)

    println("done")
  }
}
