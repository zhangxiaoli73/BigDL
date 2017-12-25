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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.nn.ReLU
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class ReLUSpec extends FlatSpec with Matchers {

  "ReLU forward" should "work correctly" in {
    val relu = ReLU[Float](ip = false)
    val input = Tensor[Float](4, 96, 55, 55).rand(-1, 1)
    val gradOutput = Tensor[Float](4, 96, 55, 55).rand(-1, 1)
    // warm up
    for (i <- 1 to 50) {
      val output = relu.forward(input)
      val gradInput = relu.backward(input, gradOutput)
    }

    val s1 = System.nanoTime()
    for (i <- 1 to 50) {
      val output = relu.forward(input)
      val gradInput = relu.backward(input, gradOutput)
    }
    val end1 = System.nanoTime() - s1
    println(s"relu time ${end1/1e9}")
    // input should be (output)
  }

  "ReLU dnn forward" should "work correctly" in {
    val relu = ReLUDnn[Float](ip = false)
    val input = Tensor[Float](4, 96, 55, 55).rand(-1, 1)
    val gradOutput = Tensor[Float](4, 96, 55, 55).rand(-1, 1)
    // warm up
    for (i <- 1 to 50) {
      val output = relu.forward(input)
      val gradInput = relu.backward(input, gradOutput)
    }

    var output : Tensor[Float] = null
    val s1 = System.nanoTime()
    for (i <- 1 to 50) {
      output = relu.forward(input)
      val gradInput = relu.backward(input, gradOutput)
    }
    val end1 = System.nanoTime() - s1
    println(s"relu dnn time ${end1/1e9}")
//    println(input)
//    println(output)
    println("done")
    // input should be (output)
  }
}
