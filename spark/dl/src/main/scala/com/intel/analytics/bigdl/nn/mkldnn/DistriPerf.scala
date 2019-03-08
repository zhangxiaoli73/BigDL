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

import java.util.concurrent.atomic.AtomicInteger

import breeze.linalg.*
import breeze.numerics._
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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, Phase}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, broadcast}
import scopt.OptionParser
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object EquivalentNew {
  def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
    val absA = math.abs(a)
    val absB = math.abs(b)
    val diff = math.abs(a - b)

    val result = if (a == b) {
      true
    } else {
      math.min(diff / (absA + absB), diff) < epsilon
    }

    result
  }

  def nearequals(t1: Tensor[Float], t2: Tensor[Float],
                 epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    result
  }
}
object DistriPerf {
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[DistriPerfParams]("BigDL w/ Dnn Local Model Performance Test") {
    opt[String]('d', "data")
      .text("model you want, vgg16 | resnet50 | vgg16_graph | resnet50_graph")
      .action((v, p) => p.copy(dataType = v))
    opt[String]('p', "path")
      .text("model you want, vgg16 | resnet50 | vgg16_graph | resnet50_graph")
      .action((v, p) => p.copy(modelPath = v))
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Boolean]('t', "threadPredict")
      .text("Whether thread predict")
      .action((v, p) => p.copy(threadPredict = v))
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)], allSum: Long): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum/1e9} realTime ${allSum/1e9} backwardSum = ${backwardSum/1e9}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long)]
    var i = 0
    while (i < times.length) {
      val rate = times(i)._2.toDouble/ allSum
      timeBuffer.append((times(i)._1, times(i)._2))
      i += 1
    }
    val sortData = timeBuffer.sortBy(a => a._2)
    println("111111111111111111        ")
    var m = 0
    while (m < sortData.length) {
      val layer = sortData(m)._1.getName()
      println(layer + "__" + sortData(m)._1 + "__" + (sortData(m)._2/1e9))
      m += 1
    }
  }

  def predict(model: Module[Float], input: MiniBatch[Float],
              params: DistriPerfParams): Unit = {
    val subModelNumber = Engine.getEngineType() match {
      case MklBlas => Engine.coreNumber()
      case MklDnn => 1
    }
    model.evaluate()
    val workingModels = if (subModelNumber != 1) {
      val wb = Util.getAndClearWeightBias(model.parameters())
      val models = (1 to subModelNumber).map(i => {
        logger.info(s"Clone $i model...")
        val m = model.cloneModule()
        Util.putWeightBias(wb, m)
        m.evaluate()
        m
      }).toArray
      Util.putWeightBias(wb, model)
      models
    } else {
      Array(model)
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
//    println(s"engine default pool size ${Engine.default.getPoolSize}")
//    val warmup = 20
//    val warmpResults = Engine.default.invoke((0 until subModelNumber).map(i =>
//      () => {
//        val localModel = workingModels(i).evaluate()
//        val data = inputBuffer(i)
//        for (i <- 0 to warmup) {
//          val output = localModel.forward(data.getInput())
//        }
//        1
//      }))
//    Engine.default.sync(warmpResults)
    println("start predict throughput test")
    val start = System.nanoTime()
    for (i <- 0 to params.iteration) {
      val results = Engine.default.invoke((0 until subModelNumber).map(i =>
        () => {
          val localModel = workingModels(i).evaluate()
          val data = inputBuffer(i)
          val s1 = System.nanoTime()
          val output = localModel.forward(data.getInput())
//          val e1 = System.nanoTime() - s1
//          getTopTimes(localModel.getTimes(), e1)
//          println(s"iteration time ${e1/1e9}")
//          localModel.resetTimes()
          1
        }))
      Engine.default.sync(results)
      // println("1111111111111111")
    }

    val end = System.nanoTime()
    logger.info(s"Use java thread ${params.modelPath} isGraph ${model.isInstanceOf[DnnGraph]} " +
      s"isIR ${model.isInstanceOf[IRGraph[Float]]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9} record / second."
    )
  }

  def dnnPredict(model: Module[Float], input: MiniBatch[Float],
              params: DistriPerfParams): Unit = {
    model.evaluate()
    println("start predict throughput test")
    val start = System.nanoTime()
    for (i <- 0 to params.iteration) {
      val s1 = System.nanoTime()
      Engine.dnnComputing.invokeAndWait2(Array(1).map(_ => () => {
        model.evaluate()
        val output = model.forward(input.getInput())
      }))
//      val e1 = System.nanoTime() - s1
//      getTopTimes(model.getTimes(), e1)
//      model.resetTimes()
//      println(s"iteration time ${e1/1e9}")
//      println("1111111111111111")
    }
    val end = System.nanoTime()
    logger.info(s"Use java thread ${params.modelPath} isGraph ${model.isInstanceOf[DnnGraph]} " +
      s"isIR ${model.isInstanceOf[IRGraph[Float]]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9} record / second."
    )
  }

  def threadPredict(params: DistriPerfParams, sc: SparkContext, modelLoad: Module[Float]): Unit = {
    val batchSize = params.batchSize
    val iterations = params.iteration

    val graph = if (!modelLoad.isInstanceOf[Graph[Float]]) modelLoad.toGraph() else modelLoad
    val model = if (Engine.getEngineType() == MklDnn) {
      val m = graph.asInstanceOf[StaticGraph[Float]].toIRgraph()
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
    if (Engine.getEngineType() == MklDnn) {
      dnnPredict(model, miniBatch, params)
    } else {
      predict(model, miniBatch, params)
    }
  }

  def accurayTest(params: DistriPerfParams, sc: SparkContext, modelLoad: Module[Float]): Unit = {
    val batchSize = params.batchSize
    val iterations = params.iteration

    val graph = if (!modelLoad.isInstanceOf[Graph[Float]]) modelLoad.toGraph() else modelLoad
    val model = if (Engine.getEngineType() == MklDnn) {
      val m = graph.asInstanceOf[StaticGraph[Float]].toIRgraph()
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

    graph.evaluate()
    model.evaluate()

    val in = Tensor[Float](inputShape).rand()

    val out2 = model.forward(in)
    val out1 = graph.forward(in)

    require(EquivalentNew.nearequals(out1.toTensor[Float], out2.toTensor[Float], 1e-6) == true)

    val tmp = 0
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriPerfParams()).foreach { params =>
      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val modelLoad = Module.loadModule[Float](params.modelPath)
      threadPredict(params, sc, modelLoad)
      // accurayTest(params, sc, modelLoad)
    }
  }
}

case class DistriPerfParams (
    batchSize: Int = 8,
    iteration: Int = 10,
    dataType: String = "ssd",
    modelPath: String = "inception_v1",
    threadPredict: Boolean = true
  )
