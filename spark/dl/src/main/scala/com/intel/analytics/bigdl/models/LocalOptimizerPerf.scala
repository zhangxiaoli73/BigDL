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
  val modelSupported = Set("inception_v1","inception_v2", "vgg16", "vgg19", "alexnet", "resnet_50",
    "lstm", "lstmpeephole", "simplernn", "gru", "convlstmpeephole")
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
    opt[Boolean]('f', "inference")
      .text("inference. One of true | false")
      .action((v, p) => p.copy(inference = v))
    help("help").text("Prints this usage text")
  }

  def getModel(module: String,
    batchSize: Int): (Module[Float], Tensor[Float], Tensor[Float], Criterion[Float]) = {
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
        val sequenceLen = 30
        val inputSize = 1280
        val hiddenSize = 128

        val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize))
        val labels = Tensor(Array(batchSize, hiddenSize)).fill(1)
        val criterion = nn.MSECriterion[Float]()


        (LSTM(1000, inputSize, hiddenSize), input, labels, criterion)

//        val model = nn.LSTM(inputSize, hiddenSize).cell
//        (model, input, labels, criterion)

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
        val labels = Tensor(Array(batchSize, sequenceLen, hiddenSize, inputWidth, inputHeight)).fill(1)
        val criterion = nn.MSECriterion[Float]()

        (ConvLSTMPeephole(1000, inputSize, hiddenSize, kernelC, kernelI, stride), input, labels, criterion)

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

      case "tanh" =>
        val model = new Tanh[Float]()
        val inputSize = 1280
        val input = Tensor[Float](batchSize, 1280).apply1(e => Random.nextFloat())
        val labels = Tensor[Float](batchSize, 1280).apply1(e => Random.nextFloat())
        val criterion = nn.MSECriterion[Float]()
        (model, input, labels, criterion)

      case "sigmoid" =>
        val model = new Sigmoid[Float]()
        val inputSize = 100
        val input = Tensor[Float](batchSize, 128).fill(1.0f)
        val labels = Tensor[Float](batchSize, 128).fill(1.0f)
        val criterion = nn.MSECriterion[Float]()
        (model, input, labels, criterion)

      case "reshape" =>
        val model = new Reshape[Float](Array(4, 128))
        val inputSize = 100
        val input = Tensor[Float](batchSize, 512).apply1(e => Random.nextFloat())
        val labels = Tensor[Float](batchSize, 4, 128).apply1(e => Random.nextFloat())
        val criterion = nn.MSECriterion[Float]()
        (model, input, labels, criterion)

      case "dropout" =>
        val model = new Reshape[Float](Array(4, 128))
        val inputSize = 100
        val input = Tensor[Float](batchSize, 512).apply1(e => Random.nextFloat())
        val labels = Tensor[Float](batchSize, 4, 128).apply1(e => Random.nextFloat())
        val criterion = nn.MSECriterion[Float]()
        (model, input, labels, criterion)

      case "lenet" =>
        def graph(classNum: Int): Module[Float] = {
          val input = Reshape(Array(1, 28, 28)).inputs()
          val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5").inputs(input)
          val tanh1 = Tanh().inputs(conv1)
          val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh1)
          val tanh2 = Tanh().inputs(pool1)
          val conv2 = SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5").inputs(tanh2)
          val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(conv2)
          val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
          val fc1 = Linear(12 * 4 * 4, 100).setName("fc1").inputs(reshape)
          val tanh3 = Tanh().inputs(fc1)
          val fc2 = Linear(100, classNum).setName("fc2").inputs(tanh3)
          val output = LogSoftMax().inputs(fc2)

          Graph(input, output)
        }
        val input = Tensor[Float](batchSize, 28, 28).apply1(e => Random.nextFloat())
        val labels = Tensor[Float](batchSize, 10).apply1(e => Random.nextFloat())
        val criterion = nn.MSECriterion[Float]()
        (graph(10), input, labels, criterion)
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

        val lossSum = Engine.default.invokeAndWait(
          (0 until param.coreNumber).map(i =>
              () => {
                // println(s"running model ${i}")
                val localModel = workingModels(i)
                localModel.zeroGradParameters()
                localModel.evaluate()
                val output = localModel.forward(inputBuffer(i))
                // localModel.backward(inputBuffer(i), output)
              })
            )
        val end = System.nanoTime()
        logger.info(s"Iteration ${i}-iteration time is ${(end - start) / 1e9}s " +
          s"Throughput is ${param.batchSize.toDouble / (end - start) * 1e9} record / second. "
          )
      }
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

      val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
        _ <: Activity, Float], Long, Long, Long, Double)]
      var i = 0
      while (i < times.length) {
        val all = times(i)._2 + times(i)._3
        val rate = times(i)._3.toDouble/ times(i)._2
        timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate))
        i += 1
      }
      val sortData = timeBuffer.sortBy(a => a._4)
      sortData.foreach(println)
    }

    def all(model: Module[Float], input: Tensor[Float]): Unit = {
      val subModelNumber = param.coreNumber
      val workingModels = (1 to param.coreNumber).map(i => {
        logger.info(s"Clone $i model...")
        model.cloneModule()
      }).toArray

      val default: ThreadPool = new ThreadPool(param.coreNumber * 50)

      val timeBuffer =
      new ArrayBuffer[(AbstractModule[_ <: Activity, _ <: Activity, Float], Long, Long, Double)]

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
              localModel.training()
              val t1 = System.nanoTime()
              val output = localModel.forward(inputBuffer(i))
              val end1 = System.nanoTime() - t1
              val t2 = System.nanoTime()
              localModel.backward(inputBuffer(i), output)
              val end2 = System.nanoTime() - t2
              val tmp = localModel.getTimes()
              getTopTimes(tmp)
              localModel.resetTimes()
              // println("forward: " + end1 + " backward: " + end2 + " rate: " + end2.toDouble/end1)
              println("forward: " + end1 + " backward: " + end2 + " rate: " + end2.toDouble/end1)
            })
        )
        val end = System.nanoTime()
        logger.info(s"Iteration ${i}-iteration time is ${(end - start) / 1e9}s " +
          s"Throughput is ${param.batchSize.toDouble / (end - start) * 1e9} record / second. "
        )
      }
    }

    Engine.setCoreNumber(param.coreNumber)

    val (_model, input, labels, criterion) = getModel(param.module, param.batchSize)
    val model = _model
    println(param.module)
    println(model)

    param.inputData match {
      case "constant" => input.toTensor[Float].fill(0.01f)
      case "random" => input.toTensor[Float].rand()
    }

//    backward(model, input)

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

    if (!param.inference) {
      val optimizer = Optimizer(model, dummyDataSet, criterion)
      optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
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
  batchSize: Int = 4,
  coreNumber: Int = 1, // Runtime.getRuntime.availableProcessors() / 2,
  iteration: Int = 80,
  dataType: String = "float",
  module: String = "lstmpeephole",
  inputData: String = "random",
  inference: Boolean = true,
  hiddenSize: Int = 128
)
