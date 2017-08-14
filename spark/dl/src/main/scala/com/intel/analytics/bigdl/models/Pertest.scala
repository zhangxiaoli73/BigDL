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
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import scopt.OptionParser

import scala.util.Random


object Pertest {
  var warmUp = 0
  var times = 1
  var seqLength = 1

  case class PerfParams(
     modelName: String = "simplernn",
     warmUp: Int = 30,
     times: Int = 200,
     batchSize: Int = 100,
     seqLength: Int = 2
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
    opt[Int]("seqLength")
      .text("seqLength")
      .action((x, c) => c.copy(seqLength = x))
  }

  def main(args: Array[String]): Unit = {
    perfParser.parse(args, new PerfParams()).map(param => {
      var name = param.modelName
      warmUp = param.warmUp
      times = param.times
      seqLength = param.seqLength

      println("model name: " + name)
      println("seqLength " + seqLength)
      println("times: " + times + " warmUp: " + warmUp)
      println("batchSize: " + param.batchSize)

      if (name == "cadd") {
         testCadd(param.batchSize)
      } else if (name == "rnn") {
          test(param.batchSize)
      } else {
        val (model, input, labels, criterion) = getModel(name, param.batchSize)
        computing(input, criterion, labels, model, name)
      }
    })
  }

  def testCadd (batchSize: Int): Unit = {
    val input1 = T(Tensor[Float](Array(batchSize, 128)).apply1(e => Random.nextFloat()),
                  Tensor[Float](batchSize, 128).apply1(e => Random.nextFloat()))
    val grad1 = Tensor(Array(batchSize, 128)).fill(1)
    val model = CAddTable[Float](false).setName("add")

    for (i <- 1 to warmUp) {
      val output = model.forward(input1)
      model.backward(input1, grad1)
    }
    val tmp = model.getTimes()
    val length = tmp.length
    var time = new Array[Double](length)
    var m = 0
    while (m < length) {
      time(m) = tmp(m)._3.toDouble / tmp(m)._2
      m += 1
    }
    var forTime : Long = 0
    var backTime : Long = 0
    val start1 = System.nanoTime()
    for (i <- 1 to times) {
      model.resetTimes()
      model.forward(input1)
      model.backward(input1, grad1)
      val tmp = model.getTimes()
      var m = 0
      while (m < length) {
        if (tmp(m)._1.getName() == "add") {
          forTime = forTime + tmp(m)._2
          backTime = backTime + tmp(m)._3
        }
        m += 1
      }
    }
    val tmp1 = model.getTimes()
    println("diff " + (backTime * 1.0/forTime) + "s" +
      " forTime: " + forTime + " backTime:" + backTime)
    println(" all takes time " + (System.nanoTime() - start1) / 1e9 + "s")

    Thread.sleep(100)
    val start4 = System.nanoTime()
    for (i <- 1 to times) {
      model.forward(input1)
      model.backward(input1, grad1)
    }
    println(" all takes time " + (System.nanoTime() - start4) / 1e9 + "s")
    Thread.sleep(100)
    val start3 = System.nanoTime()
    for (i <- 1 to times) {
      // model.forward(input1)
      model.backward(input1, grad1)
    }
    println(" only_backward takes time " + (System.nanoTime() - start3) / 1e9 + "s")
    Thread.sleep(100)
    val start2 = System.nanoTime()
    for (i <- 1 to times) {
      model.forward(input1)
    }
    println(" only_forward takes time " + (System.nanoTime() - start2) / 1e9 + "s")
  }

  def test (batchSize: Int): Unit = {
      val parallelTable = ParallelTable[Float]()
      val i2h = Linear[Float](128, 128)
      val h2h = Linear[Float](128, 128)
      parallelTable.add(i2h)
      parallelTable.add(h2h)

      val activation = Tanh[Float]()
      val cAddTable = CAddTable[Float](false).setName("add")

      val  model = Sequential[Float]()
          .add(parallelTable)
          .add(cAddTable)
          .add(activation)
          .add(ConcatTable()
            .add(Identity[Float]())
            .add(Identity[Float]())
          )

       val input1 = T(Tensor[Float](Array(batchSize, 128)).apply1(e => Random.nextFloat()),
         Tensor[Float](batchSize, 128).apply1(e => Random.nextFloat()))
      val grad1 = T(Tensor[Float](Array(batchSize, 128)).apply1(e => Random.nextFloat()),
         Tensor[Float](batchSize, 128).apply1(e => Random.nextFloat()))

      model.resetTimes()
      for (i <- 1 to warmUp) {
        val output = model.forward(input1)
        model.backward(input1, grad1)
      }
      val tmp = model.getTimes()
      val length = tmp.length
      val time = new Array[Double](length)
      var m = 0
      while (m < length) {
        if (tmp(m)._1.getName() == "add") {
          println(tmp(m)._3.toDouble / tmp(m)._2)
        }
        time(m) = tmp(m)._3.toDouble / tmp(m)._2
        m += 1
      }
      val start1 = System.nanoTime()
      var forTime : Long = 0
      var backTime : Long = 0
      for (i <- 1 to times) {
        model.resetTimes()
        model.forward(input1)
        model.backward(input1, grad1)
        val tmp = model.getTimes()
        var m = 0
        while (m < length) {
          if (tmp(m)._1.getName() == "add") {
            forTime = forTime + tmp(m)._2
            backTime = backTime + tmp(m)._3
            // println(tmp(m)._3.toDouble / tmp(m)._2)
          }
          m += 1
        }
      }

      println("diff " + (backTime * 1.0/forTime) + "s" +
        " forTime: " + forTime + " backTime:" + backTime)
      println(" all takes time " + (System.nanoTime() - start1) / 1e9 + "s")

      val start4 = System.nanoTime()
      for (i <- 1 to times) {
        model.forward(input1)
        model.backward(input1, grad1)
      }
      println(" all takes time " + (System.nanoTime() - start4) / 1e9 + "s")

      val start3 = System.nanoTime()
      for (i <- 1 to times) {
        model.backward(input1, grad1)
      }
      println(" only_backward takes time " + (System.nanoTime() - start3) / 1e9 + "s")

      val start2 = System.nanoTime()
      for (i <- 1 to times) {
        model.forward(input1)
      }
      println(" only_forward takes time " + (System.nanoTime() - start2) / 1e9 + "s")

  }

  def getModel(module: String,
               batchSize: Int): (Module[Float], Tensor[Float], Tensor[Float], Criterion[Float]) = {
    RNG.setSeed(1000)
    val (_model, input, labels, criterion) = module match {
      case "linear" =>
        val input = Tensor[Float](Array(batchSize, 128)).apply1(e => Random.nextFloat())
        val labels = Tensor(Array(batchSize, 128)).fill(1)
        val criterion = nn.MSECriterion[Float]()
        (Linear(128, 128), input, labels, criterion)
      case "inception" =>
        val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => Random.nextFloat())
        val target = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())
        val model = Inception_v1_NoAuxClassifier(1000)
        val criterion = nn.MSECriterion[Float]()
        (model, input, target, criterion)

      case "conv" =>
        val layer = new SpatialConvolution[Float](64, 192, 5, 5, 1, 1, 2, 2)
        val model = new Sequential[Float]()
        model.add(layer)
        val input = Tensor[Float](batchSize, 64, 27, 27).apply1(e => Random.nextFloat())
        val target = Tensor[Float](batchSize, 192, 27, 27).apply1(e => Random.nextFloat())
        val output = model.updateOutput(input).toTensor[Float]
        val criterion = nn.MSECriterion[Float]()
        (model, input, target, criterion)

      case "alexnet" =>
        val input = Tensor[Float](batchSize, 3, 256, 256).apply1(e => Random.nextFloat())
        val target = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())
        val criterion = nn.MSECriterion[Float]()
        val model = AlexNet(1000)
        (model, input, target, criterion)

      case "lstm" =>
        val sequenceLen = seqLength
        val inputSize = 128
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).apply1(e => Random.nextFloat())
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTM(1000, inputSize, hiddenSize), input, labels, criterion)

      case "gru" =>
        val sequenceLen = seqLength
        val inputSize = 128
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).apply1(e => Random.nextFloat())
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
        val sequenceLen = seqLength
        val inputSize = 128
        val hiddenSize = 128

        val input =
          Tensor[Float](Array(batchSize, sequenceLen, inputSize)).apply1(e => Random.nextFloat())
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (SimpleRNN(1000, inputSize, hiddenSize), input, labels, criterion)

      case "lstmpeephole" =>
        val sequenceLen = seqLength
        val inputSize = 128
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTMPeephole(1000, inputSize, hiddenSize), input, labels, criterion)

      case "tanh" =>
        val model = new Tanh[Float]()
        val input = Tensor[Float](batchSize, 128).apply1(e => Random.nextFloat())
        val labels = Tensor[Float](batchSize, 128).apply1(e => Random.nextFloat())
        val criterion = nn.MSECriterion[Float]()
        (model, input, labels, criterion)
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

    val start3 = System.nanoTime()
    for (i <- 1 to times) {
      model.backward(input, gradOutput)
    }
    println(name + " only_backward takes time " + (System.nanoTime() - start3) / 1e9 + "s")

    val start2 = System.nanoTime()
    for (i <- 1 to times) {
      model.forward(input).toTensor[Float]
    }
    println(name + " only_forward takes time " + (System.nanoTime() - start2) / 1e9 + "s")

    val start1 = System.nanoTime()
    for (i <- 1 to times) {
      model.forward(input).toTensor[Float]
      model.backward(input, gradOutput)
    }
    val tmp1 = model.getTimes()
    println(name + " all takes time " + (System.nanoTime() - start1) / 1e9 + "s")
  }
}
