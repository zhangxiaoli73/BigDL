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

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.{TensorCriterion, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._

@com.intel.analytics.bigdl.tags.Parallel
class AddSpec extends FlatSpec with Matchers {

  "A Add with scaleB" should "work correctly" in {
    val inputN = 5
    val seed = 100
    RNG.setSeed(seed)
    val layer1 = new Add[Double](inputN)
    val layer2 = layer1.cloneModule().asInstanceOf[Add[Double]]
      .setScaleB(2.0)

    val input = Tensor[Double](1, 5)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 3
    input(Array(1, 4)) = 4
    input(Array(1, 5)) = 5
    val gradOutput = Tensor[Double](5)
    gradOutput(Array(1)) = 2
    gradOutput(Array(2)) = 5
    gradOutput(Array(3)) = 10
    gradOutput(Array(4)) = 17
    gradOutput(Array(5)) = 26

    val output1 = layer1.forward(input)
    val gradInput1 = layer1.backward(input, gradOutput)
    val output2 = layer2.forward(input)
    val gradInput2 = layer2.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)

    layer2.gradBias should be (layer1.gradBias.mul(2))
  }

  "test " should "more better" in {
    val batch = 200

    val input1 = Tensor[Float](batch, 1500).randn()
    val weight1 = Tensor[Float](1500, 1).randn()
    val weight2 = Tensor[Float](1, 10000).randn()

    val output1 = Tensor[Float](batch, 1)
    val output2 = Tensor[Float](batch, 10000)

    val t11 = System.nanoTime()
    for (i <- 1 to 100) {
      output1.mm(input1, weight1)
      output2.mm(output1, weight2)
    }
    val end11 = System.nanoTime() - t11
    println("time: " + end11/1e9 + " s")

    // ************

    val input = Tensor[Float](batch, 1500).randn()
    val weight = Tensor[Float](1500, 10000).randn()

    val output = Tensor[Float](batch, 10000).rand()

    val t1 = System.nanoTime()
    for (i <- 1 to 100) {
      output.mm(input, weight)
    }
    val end1 = System.nanoTime() - t1
    println("time: " + end1/1e9 + " s")

    // ***************

  }


   "111" should "222" in {
     val inputSize = 650 // param.inputSize
     val hiddenSize = 6000 // param.hiddenSize
     val batchSize = 2

     val input = Tensor[Float](Array(batchSize, inputSize)).fill(2.0f)
     val labels = Tensor[Float](Array(batchSize, hiddenSize)).fill(1)

     RNG.setSeed(100)
     val model1 = Linear[Float](inputSize, hiddenSize)
     RNG.setSeed(100)
     val model2 = Sequential[Float]().
       add(Linear[Float](inputSize, 100)).add(Linear[Float](100, hiddenSize))

     // warm up
     for (i <- 1 to 100) {
       val out2 = model2.forward(input)
       val grad = model2.backward(input, labels)
     }
     for (i <- 1 to 100) {
       val out1 = model1.forward(input)
       val grad = model1.backward(input, labels)
     }
     // ****************
     val t1 = System.nanoTime()
     for (i <- 1 to 100) {
       val out1 = model1.forward(input)
       val grad = model1.backward(input, labels)
     }
     val end1 = System.nanoTime() - t1

     val t2 = System.nanoTime()
     for (i <- 1 to 100) {
       val out2 = model2.forward(input)
       val grad = model2.backward(input, labels)
     }
     val end2 = System.nanoTime() - t2

     println(s"end1 ${end1/1e9} end2 ${end2/1e9}")

     val grad1 = model1.getParameters()
     val grad2 = model2.getParameters()

     println("done")
  }

}
