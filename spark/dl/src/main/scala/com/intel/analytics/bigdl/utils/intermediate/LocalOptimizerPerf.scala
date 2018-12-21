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
package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, SparseMiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v2}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.util.Random
import spire.macros.Auto.scala

object LocalOptimizerPerf {
  val modelSupported = Set("inception_v1",
    "inception_v2", "vgg16", "vgg19", "alexnet", "resnet_50",
    "lstm", "lstmpeephole", "simplernn", "gru", "convlstmpeephole", "deepmf")
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
      .text("Input data type. One of constant | random | default")
      .action((v, p) => p.copy(inputData = v))
      .validate(v =>
        if (v.toLowerCase() == "constant" || v.toLowerCase() == "random" || v.toLowerCase() == "default") {
          success
        } else {
          failure("Input data type must be one of constant, random and random")
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

  def getModel(module: String,
               batchSize: Int, param: LocalOptimizerPerfParam): (Module[Float], MiniBatch[Float], Criterion[Float]) = {
    module.toLowerCase() match {
      case "alexnet" =>
        (AlexNet(1000), MiniBatch[Float](Tensor(batchSize, 3, 227, 227), Tensor(batchSize).fill(1)),
          ClassNLLCriterion[Float]())
      case "inception_v1" =>
        (Inception_v1(1000), MiniBatch[Float](Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1)),
          ClassNLLCriterion[Float]())
      case "inception_v2" =>
        (Inception_v2(1000), MiniBatch[Float](Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1)),
          ClassNLLCriterion[Float]())
      case "vgg16" =>
        (Vgg_16(1000), MiniBatch[Float](Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1)),
          ClassNLLCriterion[Float]())
      case "vgg19" =>
        (Vgg_19(1000), MiniBatch[Float](Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1)),
          ClassNLLCriterion[Float]())
      case "resnet_50" =>
        val model = ResNet(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataset" -> DatasetType.ImageNet))
        ResNet.shareGradInput(model)
        ResNet.modelInit(model)
        (model, MiniBatch(Tensor(batchSize, 3, 224, 224), Tensor(batchSize).fill(1)),
          CrossEntropyCriterion[Float]())
    }
  }

  def performance(param: LocalOptimizerPerfParam): Unit = {
    def predict(model: Module[Float], input: MiniBatch[Float]): Unit = {
      val subModelNumber = param.coreNumber
      val workingModels = {
        val wb = Util.getAndClearWeightBias(model.parameters())

        val models = (1 to subModelNumber).map(i => {
          logger.info(s"Clone $i model...")
          val m = model.cloneModule()
          Util.putWeightBias(wb, m)
          m
        }).toArray
        Util.putWeightBias(wb, model)
        models
      }

      val stackSize = input.size() / subModelNumber
      val extraSize = input.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val inputBuffer = new Array[MiniBatch[Float]](parallelism)
      var b = 0
      while (b < parallelism) {
        val offset = b * stackSize + math.min(b, extraSize) + 1
        val length = stackSize + (if (b < extraSize) 1 else 0)
        inputBuffer(b) = input.slice(offset, length)
        b += 1
      }

      val default: ThreadPool = new ThreadPool(param.coreNumber * 50)

      // warm up
      val warmup = 20
      val warmpresults = default.invoke((0 until param.coreNumber).map(i =>
        () => {
          val localModel = workingModels(i)
          val data = inputBuffer(i)
          localModel.evaluate()

          for (i <- 0 to warmup) {
            val output = localModel.forward(data.getInput()).toTensor[Float]
          }
          1
        }))
      default.sync(warmpresults)

      val start = System.nanoTime()
      val results = default.invoke((0 until param.coreNumber).map(i =>
        () => {
          val localModel = workingModels(i)
          val data = inputBuffer(i)
          localModel.evaluate()

          for (i <- 0 to param.iteration) {
            val output = localModel.forward(data.getInput()).toTensor[Float]
          }
          1
        }))
      default.sync(results)

      val end = System.nanoTime()
      logger.info(s"Iteration -iteration time is ${(end - start) / 1e9}s " +
        s"Throughput is ${param.batchSize.toDouble * param.iteration / (end - start) * 1e9} record / second. "
      )
    }

    Engine.setCoreNumber(param.coreNumber)

    val (model, miniBatch, criterion) = getModel(param.module, param.batchSize, param)
    println(model)

    val filler = param.inputData match {
      case "constant" => t: Tensor[Float] => t.fill(0.01f)
      case "random" => t: Tensor[Float] => t.rand()
      case "default" => t: Tensor[Float] => t
    }
    def fillInput(input: Activity, f: Tensor[Float] => Tensor[Float]): Unit = {
      input match {
        case t: Tensor[_] =>
          f(t.asInstanceOf[Tensor[Float]])
        case t: Table =>
          t.foreach(v => f(v._2.asInstanceOf[Tensor[Float]]))
      }
    }
    fillInput(miniBatch.getInput, filler)

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
            miniBatch
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
      fillInput(miniBatch.getInput, filler)
      predict(model, miniBatch)
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
  * @param inputData input data type (constant / random / default)
  */
case class LocalOptimizerPerfParam(
    batchSize: Int = 32,
    coreNumber: Int = Runtime.getRuntime.availableProcessors() / 2,
    iteration: Int = 50,
    dataType: String = "float",
    module: String = "lstm",
    inputData: String = "random",
    inference: Boolean = false,
    inputSize: Int = 128,
    hiddenSize: Int = 128,
    sequenceLen: Int = 30
  )