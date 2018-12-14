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
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, Phase}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, MklDnn, T, Table}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object LocalPerf {

  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[LocalPerfParams]("BigDL w/ Dnn Local Model Performance Test") {
    opt[String]('m', "model")
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
    opt[Boolean]('t', "training")
      .text(s"Perf test training or testing")
      .action((v, p) => p.copy(training = v))
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new LocalPerfParams()).foreach { params =>
      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init

      val batchSize = params.batchSize
      val training = params.training
      val iterations = params.iteration

      val inputFormat = Memory.Format.nchw
      var input : Tensor[Float] = null
      var label : Tensor[Float] = null

      val p = "/home/zhangli/workspace/zoo-model/analytics-zoo_vgg-16_imagenet_0.1.0.model"
 //    "/home/zhangli/workspace/zoo-model/analytics-zoo_frcnn-pvanet_PASCAL_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_frcnn-vgg16-compress_PASCAL_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_frcnn-vgg16_PASCAL_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_inception-v3_imagenet_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_squeezenet_imagenet_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_ssd-vgg16-300x300_PASCAL_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_ssd-vgg16-512x512_PASCAL_0.1.0.model"
//      "/home/zhangli/workspace/zoo-model/analytics-zoo_vgg-19_imagenet_0.1.0.model"
      val modelLoad = if (params.modelPath == "vgg16") {
          // Vgg_16.blas(1000)
       Vgg_16.graph(batchSize, 1000)
      } else if (params.modelPath == "vgg16_graph") {
        println("graph_type: dnn graph")
        Vgg_16.graph(batchSize, 1000)
      } else {
         Module.loadModule[Float](params.modelPath)
      }
      val graph = if (!modelLoad.isInstanceOf[Graph[Float]]) modelLoad.toGraph() else modelLoad
      val model = if (Engine.getEngineType() == MklDnn
        && params.modelPath != "vgg16" && params.modelPath != "vgg16_graph") {
        val m = graph.asInstanceOf[StaticGraph[Float]].toIRgraph(5, Memory.Format.nc)
        m.build()
        println("graph_type: dnn graph")
        m
      } else {
        graph
      }

      params.dataType match {
        case "imagenet" =>
          val inputShape = Array(batchSize, 3, 224, 224)
          input = Tensor(inputShape).rand()
          label = Tensor(batchSize).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 1000).toFloat)
        case "ssd" =>
          val inputShape = Array(batchSize, 3, 300, 300)
          input = Tensor(inputShape).rand()
          label = Tensor(batchSize).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 1000).toFloat)
        case "lenet" =>
          val inputShape = Array(batchSize, 1, 28, 28)
          input = Tensor(inputShape).rand()
          label = Tensor(batchSize).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 10).toFloat)
        case _ => throw new UnsupportedOperationException(s"Unkown model ${params.dataType}")
      }

      val criterion = CrossEntropyCriterion()

      if (training) {
        model.training()
      } else {
        if (model.isInstanceOf[DnnGraph]) {
          model.asInstanceOf[DnnGraph].compile(Phase.InferencePhase)
        }
        model.evaluate()
      }

      var iteration = 0
      val start = System.nanoTime()
      val throughputs = new ArrayBuffer[Float]()
      while (iteration < iterations) {
        val start = System.nanoTime()
        val output = model.forward(input)

        if (training) {
          val _loss = criterion.forward(output, label)
          val errors = criterion.backward(output, label).toTensor
          model.backward(input, errors)
        }

        val takes = System.nanoTime() - start
        throughputs.append(batchSize.toFloat / (takes / 1e9).toFloat)
        val throughput = "%.2f".format(batchSize.toFloat / (takes / 1e9))
        println(s"Iteration $iteration, takes $takes s, throughput is $throughput imgs/sec")

        iteration += 1
      }
      val avg = throughputs.toArray.reduce((a, b) => (a + b)) / iterations
      println(s"${params.modelPath} ${params.batchSize} Average throughput is $avg imgs/sec")
    }
  }
}

case class LocalPerfParams (
  batchSize: Int = 4,
  iteration: Int = 80,
  training: Boolean = false,
  dataType: String = "lenet",
  modelPath: String = "imagenet"
)
