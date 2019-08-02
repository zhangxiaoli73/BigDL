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

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class BoxHeadSpec extends FlatSpec with Matchers {
  "BoxHead" should "be ok" in {
    val proposals = Tensor[Float](1000, 4) // int format
    val imageInfo = Tensor[Float](2) // image size

    val features1 = Tensor[Float](1, 6, 3, 4).rand()
    val features2 = Tensor[Float](1, 6, 5, 2).rand()

  }
}
