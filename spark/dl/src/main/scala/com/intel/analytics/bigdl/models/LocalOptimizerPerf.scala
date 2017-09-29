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
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v2}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T, ThreadPool}
import com.intel.analytics.bigdl.models.{ConvLSTMPeephole, GRU, LSTM, LSTMPeephole, SimpleRNN}
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

object LocalOptimizerPerf {
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[LocalOptimizerPerfParam]("BigDL Local Performance Test") {
    head("Performance Test of Local Optimizer")
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('c', "coreNumber")
      .text("physical cores number of current machine")
      .action((v, p) => p.copy(coreNumber = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[String]('m', "model")
      .text("Model name")
      .action((v, p) => p.copy(module = v))

    opt[String]('d', "inputdata")
      .text("Input data type. One of constant | random")
      .action((v, p) => p.copy(inputData = v))
      .validate(v =>
        if (v.toLowerCase() == "constant" || v.toLowerCase() == "random") {
          success
        } else {
          failure("Input data type must be one of constant and random")
        }
      )
    opt[Boolean]('f', "inference")
      .text("inference. One of true | false")
      .action((v, p) => p.copy(inference = v))
    opt[Int]('f', "inputSize")
      .text("inputSize")
      .action((v, p) => p.copy(inputSize = v))
    opt[Int]('f', "hiddenSize")
      .text("hiddenSize")
      .action((v, p) => p.copy(hiddenSize = v))
    opt[Int]('f', "sequenceLen")
      .text("sequenceLen")
      .action((v, p) => p.copy(sequenceLen = v))
    help("help").text("Prints this usage text")
  }

  def getModel(module: String, batchSize: Int, param: LocalOptimizerPerfParam): (
    Module[Float], Tensor[Float], Tensor[Float], Criterion[Float]) = {
    val (_model, input, labels, criterion) = module match {
      case "alexnet" =>
        (AlexNet(1000), Tensor(batchSize, 3, 227, 227), Tensor(batchSize).fill(1),
          ClassNLLCriterion())
      case "inception_v1" =>
        (Inception_v1(1000), Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1),
          ClassNLLCriterion())
      case "inception_v2" =>
        (Inception_v2(1000), Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1),
          ClassNLLCriterion())
      case "vgg16" =>
        (Vgg_16(1000), Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1),
          ClassNLLCriterion())
      case "vgg19" =>
        (Vgg_19(1000), Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1),
          ClassNLLCriterion())
      case "resnet_50" =>
        val model = ResNet(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataset" -> DatasetType.ImageNet))
        ResNet.shareGradInput(model)
        ResNet.modelInit(model)

        (model, Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1), CrossEntropyCriterion())
      case "lstm" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTM(1000, inputSize, hiddenSize), input, labels, criterion)

      case "lstmpack" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTMPack(1000, inputSize, hiddenSize), input, labels, criterion)

      case "gru" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

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
        val labels = Tensor(Array(batchSize, sequenceLen, hiddenSize, inputWidth, inputHeight)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (ConvLSTMPeephole(1000, inputSize, hiddenSize, kernelC, kernelI, stride), input, labels, criterion)

      case "simplernn" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (SimpleRNN(1000, inputSize, hiddenSize), input, labels, criterion)

      case "lstmpeephole" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTMPeephole(1000, inputSize, hiddenSize), input, labels, criterion)
    }
    (_model, input, labels, criterion)
  }

  def performance(param: LocalOptimizerPerfParam): Unit = {
    def predict(model: Module[Float], input: Tensor[Float]): Unit = {
      val subModelNumber = param.coreNumber
      val workingModels = (1 to param.coreNumber).map(i => {
        logger.info(s"Clone $i model...")
        model.cloneModule()
      }).toArray

      val stackSize = input.size(1) / subModelNumber
      val extraSize = input.size(1) % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val inputBuffer = new Array[Tensor[Float]](parallelism)
      var b = 0
      while (b < parallelism) {
        val offset = b * stackSize + math.min(b, extraSize) + 1
        val length = stackSize + (if (b < extraSize) 1 else 0)
        inputBuffer(b) = input.narrow(1, offset, length)
        b += 1
      }

      val default: ThreadPool = new ThreadPool(param.coreNumber * 50)

      //warm up
      val warmup = 20
      val warmpresults = default.invoke((0 until param.coreNumber).map(i =>
        () =>  {
          val localModel = workingModels(i)
          val data = inputBuffer(i)
          localModel.zeroGradParameters()
          localModel.evaluate()

          for (i <-0 to warmup) {
            val output = localModel.forward(data).toTensor[Float]
          }
          1
        }))
      default.sync(warmpresults)

      val start = System.nanoTime()
      val results = default.invoke((0 until param.coreNumber).map(i =>
        () =>  {
          val localModel = workingModels(i)
          val data = inputBuffer(i)
          localModel.zeroGradParameters()
          localModel.evaluate()

          for (i <-0 to param.iteration) {
            val output = localModel.forward(data).toTensor[Float]
          }
          1
        }))
      default.sync(results)

      val end = System.nanoTime()
      logger.info(s"Iteration -iteration time is ${(end - start) / 1e9}s " +
        s"Throughput is ${param.batchSize.toDouble * param.iteration / (end - start) * 1e9} record / second. "
      )
    }

    def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
      Long, Long)]): Unit = {
      var forwardSum = 0L
      var backwardSum = 0L
      times.foreach(x => {
        forwardSum += x._2
        backwardSum += x._3
      })
      println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}")

      val allSum = forwardSum + backwardSum

      val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
        _ <: Activity, Float], Long, Long, Long, Double, Double)]
      var i = 0
      while (i < times.length) {
        val all = times(i)._2 + times(i)._3
        val rate = times(i)._3.toDouble/ times(i)._2
        val rate2 = all.toDouble/allSum
        if (rate2 > 0.01) {
          timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate, rate2))
        }
        i += 1
      }
      val sortData = timeBuffer.sortBy(a => a._4)
      sortData.foreach(println)
      println("\n")
    }

    def time(model: Module[Float], input: Tensor[Float], labels: Tensor[Float]): Unit = {
      // warm up
      for (i <- 1 to 20) {
        model.forward(input)
        ///model.backward(input, labels)
      }
      println("start")
      val s1 = System.nanoTime()
      for (i <- 1 to 100) {
        model.forward(input)
        ///model.backward(input, labels)
      }
      val end1 = System.nanoTime() - s1
      println(s"all time ${end1/1e9}")

      println("time for each layer")
      for (i <- 1 to 6) {
        model.resetTimes()
        model.forward(input)
        ///model.backward(input, labels)
        getTopTimes(model.getTimes())
        model.resetTimes()
      }
    }

    Engine.setCoreNumber(param.coreNumber)

    val (_model, input, labels, criterion) = getModel(param.module, param.batchSize, param)
    val model = _model
    println(model)

    param.inputData match {
      case "constant" => input.fill(0.01f)
      case "random" => input.rand()
    }

    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          private val index = new AtomicInteger()
          override def hasNext: Boolean = {
            if (train) {
              true
            } else {
              index.get() < 100000
            }
          }

          override def next(): MiniBatch[Float] = {
            index.getAndIncrement()
            MiniBatch(input, labels)
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    if (!param.inference) {
      val optimizer = Optimizer(model, dummyDataSet, criterion)
      optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
    } else {
      time(model, input, labels)
    }
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new LocalOptimizerPerfParam()).foreach(performance)
  }

}

/**
 * Local Optimizer Performance Parameters
 *
 * @param batchSize batch size
 * @param coreNumber core number
 * @param iteration how many iterations to run
 * @param dataType data type (double / float)
 * @param module module name
 * @param inputData input data type (constant / random)
 */
case class LocalOptimizerPerfParam(
  batchSize: Int = 20,
  coreNumber: Int = Runtime.getRuntime.availableProcessors() / 2,
  iteration: Int = 50,
  dataType: String = "float",
  module: String = "lstmpack",
  inputData: String = "random",
  inference: Boolean = true,
  inputSize: Int = 650,
  hiddenSize: Int = 650,
  sequenceLen: Int = 3
)
