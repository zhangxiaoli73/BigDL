/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.NarrowTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class NarrowTableSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A NarrowTable Module " should "generate correct output and grad" in {
    val module = new NarrowTable[Double](1)

    val input = T()
    input(1.0) = Tensor[Double](2, 3).apply1(e => Random.nextDouble())
    input(2.0) = Tensor[Double](2, 1).apply1(e => Random.nextDouble())
    input(3.0) = Tensor[Double](2, 2).apply1(e => Random.nextDouble())

    val gradOutput = T()
    gradOutput(1.0) = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    gradOutput(2.0) = Tensor[Double](2, 5).apply1(e => Random.nextDouble())

    val code = "module = nn.NarrowTable(1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Table]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : NarrowTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
