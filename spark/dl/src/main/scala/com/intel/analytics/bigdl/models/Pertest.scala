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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.example.loadmodel.{AlexNet, AlexNet_OWT}
import com.intel.analytics.bigdl.models.autoencoder.Autoencoder
import com.intel.analytics.bigdl.models.inception.{Inception_Layer_v1, Inception_v1, Inception_v1_NoAuxClassifier, Inception_v2}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.lenet.Utils.{TrainParams, _}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.vgg.{VggForCifar10, Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import scopt.OptionParser


object Pertest {
  var warmUp = 0
  var times = 1

  case class PerfParams(
     modelName: String = "simplernn",
     warmUp: Int = 15,
     times: Int = 200,
     batchSize: Int = 100
   )

  val perfParser = new OptionParser[PerfParams]("BigDL Perf Tests") {
    opt[String]('n', "modelName")
      .text("modelName")
      .action((x, c) => c.copy(modelName = x))
    opt[Int]("warmUp")
      .text("warmUp")
      .action((x, c) => c.copy(warmUp = x))
    opt[Int]("times")
      .text("times")
      .action((x, c) => c.copy(times = x))
    opt[Int]("batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x))
  }

  def main(args: Array[String]): Unit = {
    perfParser.parse(args, new PerfParams()).map(param => {
      val name = param.modelName
      warmUp = param.warmUp
      times = param.times

      println("model name: " + name)
      println("times: " + times + " warmUp: " + warmUp)
      println("batchSize: " + param.batchSize)

      val (model, input, labels, criterion) = getModel(name, param.batchSize)
      computing(input, criterion, labels, model, name)
    })
  }

  def getModel(module: String,
               batchSize: Int): (Module[Float], Tensor[Float], Tensor[Float], Criterion[Float]) = {
    RNG.setSeed(1000)
    val (_model, input, labels, criterion) = module match {
      case "lstm" =>
        val sequenceLen = 30
        val inputSize = 128
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTM(1000, inputSize, hiddenSize), input, labels, criterion)

      case "gru" =>
        val sequenceLen = 30
        val inputSize = 128
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (GRU(1000, inputSize, hiddenSize), input, labels, criterion)

      case "convlstmpeephole" =>
        val sequenceLen = 3
        val inputSize = 3
        val hiddenSize = 128
        val kernelC = 3
        val kernelI = 3
        val stride = 1

        val (inputWidth, inputHeight) = (112, 112)
        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize, inputWidth, inputHeight))
        val labels =
          Tensor(Array(batchSize, sequenceLen, hiddenSize, inputWidth, inputHeight)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (ConvLSTMPeephole(1000, inputSize, hiddenSize, kernelC, kernelI, stride),
          input, labels, criterion)

      case "simplernn" =>
        val sequenceLen = 30
        val inputSize = 128
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (SimpleRNN(1000, inputSize, hiddenSize), input, labels, criterion)

      case "lstmpeephole" =>
        val sequenceLen = 30
        val inputSize = 128
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTMPeephole(1000, inputSize, hiddenSize), input, labels, criterion)
    }
    (_model, input, labels, criterion)
  }

  def computing(input: Tensor[Float], criterion: Criterion[Float], target: Tensor[Float],
                model: Module[Float], name: String = ""): Unit = {
    // warm up
    var gradOutput: Tensor[Float] = null
    for (i <- 1 to warmUp) {
      val output = model.forward(input).toTensor[Float]
      val loss = criterion.forward(output, target)
      gradOutput = criterion.backward(output, target).toTensor[Float]
      model.backward(input, gradOutput)
    }
    val start1 = System.nanoTime()
    for (i <- 1 to times) {
      model.forward(input).toTensor[Float]
      model.backward(input, gradOutput)
    }
    println(name + " all takes time " + (System.nanoTime() - start1) / 1e9 + "s")

    val start2 = System.nanoTime()
    for (i <- 1 to times) {
      model.forward(input).toTensor[Float]

    }
    println(name + " only_forward takes time " + (System.nanoTime() - start2) / 1e9 + "s")
  }
}