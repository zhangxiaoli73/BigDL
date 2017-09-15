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

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v2}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.rnn.PTBModel
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T, ThreadPool}
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

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
      .text(s"Model name. ")
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
    opt[String]('f', "testType")
      .text("inference. One of true | false")
      .action((v, p) => p.copy(testType = v))
    opt[Int]('f', "inputSize")
      .text("inference. One of true | false")
      .action((v, p) => p.copy(inputSize = v))
   opt[Int]('f', "hiddenSize")
      .text("inference. One of true | false")
      .action((v, p) => p.copy(hiddenSize = v))
    opt[String]('m', "modelType")
      .text(s"Model type. ")
      .action((v, p) => p.copy(modelType = v))
    help("help").text("Prints this usage text")
  }

  def getModel(module: String, batchSize: Int, param: LocalOptimizerPerfParam): (
    Module[Float], Tensor[Float], Tensor[Float], Criterion[Float]) = {
    val (_model, input, labels, criterion) = module match {
      case "lstm" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).randn()
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTMPerf(1000, inputSize, hiddenSize), input, labels, criterion)

      case "lstm_new" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).randn()
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTMPerfNew(1000, inputSize, hiddenSize), input, labels, criterion)

      case "gru" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).randn()
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (GRUPerf(1000, inputSize, hiddenSize), input, labels, criterion)

      case "simplernn" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).randn()
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (SimpleRNNPerf(1000, inputSize, hiddenSize), input, labels, criterion)

      case "simplernn_new" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).randn()
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (SimpleRNNPerfNew(1000, inputSize, hiddenSize), input, labels, criterion)

      case "lstmpeephole" =>
        val sequenceLen = param.sequenceLen
        val inputSize = param.inputSize
        val hiddenSize = param.hiddenSize

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).randn()
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (LSTMPeepholePerf(1000, inputSize, hiddenSize), input, labels, criterion)

      case "ptb" =>
        val (sequenceLen, inputSize, hiddenSize, numLayers) =
        if (param.modelType == "small") {
          (20, 10000, 200, 2)
        } else if (param.modelType == "meduim") {
          (35, 10000, 650, 2)
        } else if (param.modelType == "large") {
          (35, 10000, 1500, 2)
        } else {
          throw new IllegalArgumentException(s"wrong model type ${param.modelType}")
        }

        val input = Tensor[Float](param.batchSize, sequenceLen).apply1(e => Random.nextFloat()*100 + 10)
        val labels = Tensor(param.batchSize, sequenceLen).fill(100.0f)
        val criterion = nn.ClassNLLCriterion[Float]()

        val model = PTBModel(inputSize = inputSize,
          hiddenSize = hiddenSize,
          outputSize = inputSize,
          numLayers = numLayers)

        model.reset()
        (model, input, labels, criterion)
    }
    (_model, input, labels, criterion)
  }

  def performance(param: LocalOptimizerPerfParam): Unit = {
    def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
      Long, Long)], totalTime: Long): Unit = {
      var forwardSum = 0L
      var backwardSum = 0L
      times.foreach(x => {
        forwardSum += x._2
        backwardSum += x._3
      })
      println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}",
        s"whole time = ${totalTime}")

      val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
        _ <: Activity, Float], Long, Long, Long, Double, Double)]
      var i = 0
      while (i < times.length) {
        val all = times(i)._2 + times(i)._3
        val rate = times(i)._3.toDouble/ times(i)._2
        val rateofAll = all.toDouble/totalTime
        timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate, rateofAll))
        i += 1
      }
      val sortData = timeBuffer.sortBy(a => a._4)
      sortData.foreach(println)
    }

    def predict(model: Module[Float], input: Tensor[Float]): Unit = {
      val subModelNumber = param.coreNumber
      val workingModels = (1 to param.coreNumber).map(i => {
        logger.info(s"Clone $i model...")
        model.cloneModule()
      }).toArray

      val default: ThreadPool = new ThreadPool(param.coreNumber * 50)

      for (i <- 0 to param.iteration) {
        val start = System.nanoTime()

        var b = 0
        val stackSize = input.size(1) / subModelNumber
        val extraSize = input.size(1) % subModelNumber
        val parallelism = if (stackSize == 0) extraSize else subModelNumber
        val inputBuffer = new Array[Tensor[Float]](parallelism)
        while (b < parallelism) {
          val offset = b * stackSize + math.min(b, extraSize) + 1
          val length = stackSize + (if (b < extraSize) 1 else 0)
          inputBuffer(b) = input.narrow(1, offset, length)
          b += 1
        }

        val lossSum = default.invokeAndWait(
          (0 until param.coreNumber).map(i =>
              () => {
                val localModel = workingModels(i)
                localModel.zeroGradParameters()
                localModel.evaluate()
//                val t1 = System.nanoTime()
                localModel.forward(inputBuffer(i))
//                val end1 = System.nanoTime() - t1
//                println("forward: " + end1/1e9)
              })
            )
        val end = System.nanoTime()
        logger.info(s"Iteration ${i}-iteration time is ${(end - start) / 1e9}s " +
          s"Throughput is ${param.batchSize.toDouble / (end - start) * 1e9} record / second. "
          )
      }
    }

    def times(model: Module[Float], input: Tensor[Float]): Unit = {
      val subModelNumber = param.coreNumber
      val workingModels = (1 to param.coreNumber).map(i => {
        logger.info(s"Clone $i model...")
        model.cloneModule()
      }).toArray

      val default: ThreadPool = new ThreadPool(param.coreNumber * 50)

      for (i <- 0 to param.iteration) {
        val start = System.nanoTime()

        var b = 0
        val stackSize = input.size(1) / subModelNumber
        val extraSize = input.size(1) % subModelNumber
        val parallelism = if (stackSize == 0) extraSize else subModelNumber
        val inputBuffer = new Array[Tensor[Float]](parallelism)
        while (b < parallelism) {
          val offset = b * stackSize + math.min(b, extraSize) + 1
          val length = stackSize + (if (b < extraSize) 1 else 0)
          inputBuffer(b) = input.narrow(1, offset, length)
          b += 1
        }

        val lossSum = default.invokeAndWait(
          (0 until param.coreNumber).map(i =>
            () => {
              // println(s"running model ${i}")
              val localModel = workingModels(i)
              localModel.zeroGradParameters()
              localModel.evaluate()
              val t1 = System.nanoTime()
              val output = localModel.forward(inputBuffer(i))
              localModel.backward(inputBuffer(i), output)
              val end = System.nanoTime() - t1
              val timeData = localModel.getTimes()
              localModel.resetTimes()
              getTopTimes(timeData, end)
            })
        )
        val end = System.nanoTime()
        logger.info(s"Iteration ${i}-iteration time is ${(end - start) / 1e9}s " +
          s"Throughput is ${param.batchSize.toDouble / (end - start) * 1e9} record / second. "
        )
      }
    }

    def all(model: Module[Float], input: Tensor[Float]): Unit = {
      val subModelNumber = param.coreNumber
      val workingModels = (1 to param.coreNumber).map(i => {
        logger.info(s"Clone $i model...")
        model.cloneModule()
      }).toArray

      val default: ThreadPool = new ThreadPool(param.coreNumber * 50)

      for (i <- 0 to param.iteration) {
        val start = System.nanoTime()

        var b = 0
        val stackSize = input.size(1) / subModelNumber
        val extraSize = input.size(1) % subModelNumber
        val parallelism = if (stackSize == 0) extraSize else subModelNumber
        val inputBuffer = new Array[Tensor[Float]](parallelism)
        while (b < parallelism) {
          val offset = b * stackSize + math.min(b, extraSize) + 1
          val length = stackSize + (if (b < extraSize) 1 else 0)
          inputBuffer(b) = input.narrow(1, offset, length)
          b += 1
        }

        val lossSum = default.invokeAndWait(
          (0 until param.coreNumber).map(i =>
            () => {
              // println(s"running model ${i}")
              val localModel = workingModels(i)
              localModel.zeroGradParameters()
              localModel.evaluate()
              val t1 = System.nanoTime()
              val output = localModel.forward(inputBuffer(i))
              val end1 = System.nanoTime() - t1
              val t2 = System.nanoTime()
              localModel.backward(inputBuffer(i), output)
              val end2 = System.nanoTime() - t2
              println("forward: " + end1/1e9 + " backward: "
                + end2/1e9 + " rate: " + end2.toDouble/end1)
            })
        )
        val end = System.nanoTime()
        logger.info(s"Iteration ${i}-iteration time is ${(end - start) / 1e9}s " +
          s"Throughput is ${param.batchSize.toDouble / (end - start) * 1e9} record / second. "
        )
      }
    }

    Engine.setCoreNumber(param.coreNumber)

    val (_model, input, labels, criterion) = getModel(param.module, param.batchSize, param)
    val model = _model
    println(model)
    println(param.module + " : " + param.testType)

//    param.inputData match {
//      case "constant" => input.fill(0.01f)
//      case "random" => input.rand()
//    }

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

    if (param.testType == "train") {
      val optimizer = Optimizer(model, dummyDataSet, criterion)
      optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
    } else if (param.testType == "predict") {
      predict(model, input)
    } else if (param.testType == "times") {
      param.coreNumber = 1
      times(model, input)
    } else {
      all(model, input)
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
  batchSize: Int = 80,
  var coreNumber: Int = Runtime.getRuntime.availableProcessors() / 2,
  iteration: Int = 30,
  dataType: String = "float",
  module: String = "lstm",
  inputData: String = "random",
  testType: String = "times",
  modelType: String = "small",
  inputSize: Int = 1500,
  hiddenSize: Int = 1500,
  sequenceLen: Int = 30
)
