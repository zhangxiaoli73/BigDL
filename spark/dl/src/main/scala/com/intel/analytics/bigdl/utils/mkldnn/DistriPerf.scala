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

package com.intel.analytics.bigdl.utils.mkldnn

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.{Module, utils}
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.{Graph, Module, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, Phase}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, broadcast}
import scopt.OptionParser
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object DistriPerf {

  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[DistriPerfParams]("BigDL w/ Dnn Local Model Performance Test") {
    opt[String]('m', "model")
      .text("model you want, vgg16 | resnet50 | vgg16_graph | resnet50_graph")
      .action((v, p) => p.copy(dataType = v))
    opt[String]('p', "path")
      .text("model you want, vgg16 | resnet50 | vgg16_graph | resnet50_graph")
      .action((v, p) => p.copy(modelPath = v))
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[String]('t', "modelType")
      .text("modelType")
      .action((v, p) => p.copy(modelType = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
  }

  def testConv(shape: Array[Int]) : Module[Float] = {
    import com.intel.analytics.bigdl.nn.mkldnn
    val input = mkldnn.Input(shape, Memory.Format.nchw).inputs()
    val conv = mkldnn.SpatialConvolution(512, 126, 3, 3, 1, 1).inputs(input)
    val out = mkldnn.Output(Memory.Format.nchw).inputs(conv)
    DnnGraph(Array(input), Array(out))
  }

  def predict(model: Module[Float], input: MiniBatch[Float],
              params: DistriPerfParams): Unit = {
    val subModelNumber = Engine.getEngineType() match {
      case MklBlas => Engine.coreNumber()
      case MklDnn => 1
    }
    val workingModels = if (subModelNumber != 1) {
      val wb = Util.getAndClearWeightBias(model.parameters())
      val models = (1 to subModelNumber).map(i => {
        logger.info(s"Clone $i model...")
        val m = model.cloneModule()
        Util.putWeightBias(wb, m)
        m
      }).toArray
      Util.putWeightBias(wb, model)
      models
    } else {
      Array(model)
    }

    if (model.isInstanceOf[DnnGraph]) {
      workingModels.foreach(model => model.asInstanceOf[DnnGraph].compile(Phase.InferencePhase))
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

    // warm up
    val warmup = 20
    val warmpResults = Engine.default.invoke((0 until subModelNumber).map(i =>
      () => {
        val localModel = workingModels(i)
        val data = inputBuffer(i)
        if (params.modelType != "resnet50") {
          localModel.evaluate()
        }
        for (i <- 0 to warmup) {
          val output = localModel.forward(data.getInput()).toTensor[Float]
        }
        1
      }))
    Engine.default.sync(warmpResults)

    println("start predict throughput test")
    val start = System.nanoTime()
    val results = Engine.default.invoke((0 until subModelNumber).map(i =>
      () => {
        val localModel = workingModels(i)
        val data = inputBuffer(i)
        if (params.modelType != "resnet50") {
          localModel.evaluate()
        }
        for (i <- 0 to params.iteration) {
          val output = localModel.forward(data.getInput()).toTensor[Float]
        }
        1
      }))
    Engine.default.sync(results)

    val end = System.nanoTime()
    logger.info(s"${params.modelPath} isGraph ${model.isInstanceOf[DnnGraph]} " +
      s"isIR ${model.isInstanceOf[IRGraph[Float]]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " +
      s"Average Throughput is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9} record / second."
    )
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriPerfParams()).foreach { params =>
      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = params.batchSize
      val iterations = params.iteration

      val inputFormat = Memory.Format.nchw
      var input : Tensor[Float] = null
      var label : Tensor[Float] = null

      val modelLoad = if (params.modelPath == "vgg16") {
        import com.intel.analytics.bigdl.models.vgg
        vgg.Vgg_16(1000, false)
      } else if (params.modelPath == "vgg19") {
        import com.intel.analytics.bigdl.models.vgg
        vgg.Vgg_19(1000, false)
      } else if (params.modelPath == "resnet50") {
        ResNet(1000, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
          "optnet" -> false, "dataSet" -> DatasetType.ImageNet))
      } else {
        Module.loadModule[Float](params.modelPath)
      }

      val graph = if (!modelLoad.isInstanceOf[Graph[Float]]) modelLoad.toGraph() else modelLoad
      val model = if (Engine.getEngineType() == MklDnn) {
        val m = graph.asInstanceOf[StaticGraph[Float]].toIRgraph(5, Memory.Format.nc)
        m.build()
        println("graph_type: dnn graph")
        m
      } else {
        graph
      }

      var inputShape: Array[Int] = null
      var outputShape: Array[Int] = null
      params.dataType match {
        case "imagenet" =>
          inputShape = Array(batchSize, 3, 224, 224)
          outputShape = Array(batchSize)
        case "ssd" =>
          inputShape = Array(batchSize, 3, 300, 300)
          outputShape = Array(batchSize)
        case "lenet" =>
          inputShape = Array(batchSize, 1, 28, 28)
          outputShape = Array(batchSize)
        case "conv" =>
          inputShape = Array(batchSize, 512, 10, 10)
          outputShape = Array(batchSize)
        case _ => throw new UnsupportedOperationException(s"Unkown model ${params.dataType}")
      }

      val miniBatch = MiniBatch(Tensor(inputShape).rand(), Tensor(outputShape).rand())
      predict(model, miniBatch, params)
    }
  }
}

case class DistriPerfParams (
  batchSize: Int = 4,
  iteration: Int = 80,
  dataType: String = "ssd",
  modelPath: String = "vgg16",
  modelType: String = "mkldnn"
)
