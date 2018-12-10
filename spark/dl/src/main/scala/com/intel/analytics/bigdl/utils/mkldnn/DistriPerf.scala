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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.{Graph, Module, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, Phase}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, broadcast}
import scopt.OptionParser
import com.intel.analytics.bigdl.utils.MklDnn

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

//      val modelLoad = if (params.modelPath == "lenet") {
//        LeNet5.graph(10)
//      } else if (params.modelPath == "inceptionV1") {
//        Inception_v1_NoAuxClassifier.graph(1000)
//      } else if (params.modelPath == "testConv") {
//        testConv(Array(batchSize, 512, 10, 10))
//      } else {
//        Module.loadModule[Float](params.modelPath)
//      }

      val modelLoad = testConv(Array(batchSize, 512, 10, 10))
      modelLoad.asInstanceOf[DnnGraph].compile(Phase.InferencePhase)

      val graph = if (!modelLoad.isInstanceOf[Graph[Float]]) modelLoad.toGraph() else modelLoad
      val model = if (params.modelType == "mkldnn") {
        val m = graph.asInstanceOf[StaticGraph[Float]].toIRgraph(5, Memory.Format.nc)
        m.build()
        m
      } else {
        graph
      }

      var inputShape: Array[Int] = null
      var outputShape: Array[Int] = null
      params.dataType match {
        case "imagenet" =>
          inputShape = Array(3, 224, 224)
          outputShape = Array(1)
        case "ssd" =>
          inputShape = Array(3, 300, 300)
          outputShape = Array(1)
        case "lenet" =>
          inputShape = Array(1, 28, 28)
          outputShape = Array(1)
        case "conv" =>
          inputShape = Array(512, 10, 10)
          outputShape = Array(1)
        case _ => throw new UnsupportedOperationException(s"Unkown model ${params.dataType}")
      }

      model.evaluate()
      val broadcast = sc.broadcast(inputShape, outputShape)
      val length = batchSize * iterations
      val rdd = sc.parallelize((1 to length), Engine.nodeNumber)
        .mapPartitions(iter => {
          val broad = broadcast.value
          val inputShape = broad._1
          val outputShape = broad._2
          val t = new ArrayBuffer[Sample[Float]]()
          while (iter.hasNext) {
            iter.next()
            val input = Tensor(inputShape).rand()
            val label = Tensor(outputShape).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 1000).toFloat)
            t.append(Sample(input, label))
          }
          t.toIterator
        }).persist()

      // engine env set
      val nExecutor = Engine.nodeNumber()
      val executorCores = Engine.coreNumber()
      Engine.setNodeAndCore(nExecutor, executorCores)
      logger.info("model thread pool size is " + Engine.model.getPoolSize)

      val start = System.nanoTime()
      val result = model.evaluate(rdd,
        Array(new Top1Accuracy[Float]), Some(batchSize))
      val takes = System.nanoTime() - start
      val avgTime = takes / 1e9
      val avg = length / avgTime
      logger.info(s"Average throughput is $avg imgs/sec")
      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    }
  }
}

case class DistriPerfParams (
  batchSize: Int = 4,
  iteration: Int = 80,
  dataType: String = "conv",
  modelPath: String = "lenet",
  modelType: String = "conv"
)
