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

import com.intel.analytics.bigdl.tensor.Tensor
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object MaskUtils {
  def loadWeight(path: String, size: Array[Int]): Tensor[Float] = {
    val weight = Tensor[Float](size)
    val arr = weight.storage().array()

    var totalLength = 0
    for (line <- Source.fromFile(path).getLines) {
      val ss = line.split(",")
      for (i <- 0 to ss.length - 1) {
        if (ss(i) != "") {
          arr(totalLength) = ss(i).toFloat
          totalLength += 1
        }
      }
    }

    require(totalLength == size.product, s"total length ${totalLength} ${size.product}")
    weight
  }


}
