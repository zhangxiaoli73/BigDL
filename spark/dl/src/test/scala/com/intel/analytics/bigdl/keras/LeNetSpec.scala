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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.{DnnGraph, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{DnnTools, HeapData}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

class LeNetSpec extends FlatSpec with Matchers {

  "LeNet sequential" should "generate the correct outputShape" in {
    val lenet = LeNet5.keras(classNum = 10)
    lenet.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }

  "LeNet graph" should "generate the correct outputShape" in {
    val lenet = LeNet5.kerasGraph(classNum = 10)
    lenet.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }

  "LeNet sequential forward and backward" should "work properly" in {
    val lenet = LeNet5.keras(classNum = 10)
    val input = Tensor[Float](Array(2, 28, 28, 1)).rand()
    val output = lenet.forward(input)
    val gradInput = lenet.backward(input, output)
  }

  "LeNet graph forward and backward" should "work properly" in {
    val lenet = LeNet5.kerasGraph(classNum = 10)
    val input = Tensor[Float](Array(2, 28, 28, 1)).rand()
    val output = lenet.forward(input)
    val gradInput = lenet.backward(input, output)
  }

  "LeNet forward with incompatible input tensor" should "raise an exception" in {
    intercept[RuntimeException] {
      val lenet = LeNet5.keras(classNum = 10)
      val input = Tensor[Float](Array(28, 28, 1)).rand()
      val output = lenet.forward(input)
    }
  }

  "Dnn LeNet to graph" should "work properly" in {
    val batchSize = 4
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 1, 28, 28)

    RNG.setSeed(seed)
    val lenet = LeNet5.dnn(batchSize, classNum = 10)
    RNG.setSeed(seed)
    val lenet2 = LeNet5.dnn(batchSize, classNum = 10)

    lenet.compile(TrainingPhase)
    lenet2.compile(TrainingPhase)

    val m = lenet2.toGraph()

    val input = Tensor[Float](Array(batchSize, 1, 28, 28)).rand() // fill(1.0f)
    val gradOutput = Tensor[Float](Array(batchSize, 10)).rand() // fill(3.0f)

    val output = DnnTools.dense(lenet.forward(input))
    val gradInput = DnnTools.dense(lenet.backward(input, gradOutput))

    val output2 = DnnTools.dense(m.forward(input))
    // val g = m.backward(input, gradOutput)

    val gradInput2 = DnnTools.dense(m.backward(input, gradOutput))

    output should be(output2)
    gradInput should be(gradInput2)

    val t = 0
  }

  "Graph LeNet to dnn graph LeNet" should "work properly" in {
    type Module[T] =
    com.intel.analytics.bigdl.nn.abstractnn.AbstractModule[Activity, Activity, T]

    System.setProperty("bigdl.mkldnn.fusion.convrelu", "fals")

    val batchSize = 4
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 1, 28, 28)

    RNG.setSeed(seed)
    val lenet = LeNet5.graphTest(batchSize)
    RNG.setSeed(seed)
    val dnn = LeNet5.dnn1(batchSize)
    RNG.setSeed(seed)
    val lenetPre = LeNet5.graphTest(batchSize)
    val lenetDnn = lenetPre.toDnnModule().toGraph()

    lenetDnn.asInstanceOf[DnnGraph[Float]]
      .compile(TrainingPhase, Array(HeapData(inputShape, inputFormat)))

    dnn.compile(TrainingPhase, Array(HeapData(inputShape, inputFormat)))

    val input = Tensor[Float](Array(batchSize, 1, 28, 28)).fill(0.1f) // rand()// rand()
    val output = lenet.forward(input).toTensor[Float]

    val gradOutput = Tensor[Float]().resizeAs(output).fill(1f)

    val gradInput = lenet.backward(input, gradOutput).toTensor[Float]
    val outputDnn = DnnTools.dense(
      lenetDnn.asInstanceOf[Module[Float]].forward(input)).toTensor[Float]
    val gradInputDnn = DnnTools.dense(
      lenetDnn.asInstanceOf[Module[Float]].backward(input, gradOutput)).toTensor[Float]


    val outputDnn1 = DnnTools.dense(
      dnn.asInstanceOf[Module[Float]].forward(input)).toTensor[Float]
    val gradInputDnn1 = DnnTools.dense(
      dnn.asInstanceOf[Module[Float]].backward(input, gradOutput)).toTensor[Float]

//    println(gradInput)
//    println(gradInputDnn)
    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val t = 0
  }

}
