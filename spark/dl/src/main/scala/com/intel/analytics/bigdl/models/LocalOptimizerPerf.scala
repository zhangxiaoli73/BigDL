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
import com.intel.analytics.bigdl.models.rnn.PTBModel
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T, ThreadPool}
import org.apache.log4j.Logger
import scopt.OptionParser

object LocalOptimizerPerf {
  val modelSupported = Set("inception_v1","inception_v2", "vgg16", "vgg19", "alexnet", "resnet_50",
    "lstm", "lstmpeephole", "simplernn", "gru", "convlstmpeephole", "ptb")
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
      .text(s"Model name. It can be ${modelSupported.mkString("| ")}")
      .action((v, p) => p.copy(module = v))
      .validate(v =>
        if (modelSupported.contains(v.toLowerCase())) {
          success
        } else {
          failure(s"Model name only supports ${modelSupported.mkString(" | ")}")
        }
      )
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

      case "ptb" =>
        val sequenceLen = 20 // param.sequenceLen
        val inputSize = 10001 // param.inputSize
        val hiddenSize = 200 // param.hiddenSize
        val numLayers = 2

        val input = Tensor[Float](Array(param.batchSize, 20)).fill(100.0f)
        val labels = Tensor(Array(param.batchSize, 20)).fill(100.0f)
        val criterion = nn.MSECriterion[Float]()

        val model = PTBModel(inputSize = inputSize,
          hiddenSize = hiddenSize,
          outputSize = inputSize,
          numLayers = numLayers)

        (model, input, labels, criterion)
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

    param.inputData match {
      case "constant" => input.fill(0.01f)
      case "random" => input.rand()
    }

    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          override def hasNext: Boolean = true

          override def next(): MiniBatch[Float] = {
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
  batchSize: Int = 32,
  coreNumber: Int = Runtime.getRuntime.availableProcessors() / 2,
  iteration: Int = 80,
  dataType: String = "float",
  module: String = "lstm",
  inputData: String = "random",
  testType: String = "all",
  inputSize: Int = 1000,
  hiddenSize: Int = 200,
  sequenceLen: Int = 30
)
