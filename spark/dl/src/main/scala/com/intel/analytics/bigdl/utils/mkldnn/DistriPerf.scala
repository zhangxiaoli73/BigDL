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
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.vgg.Vgg_16
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import scopt.OptionParser

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
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriPerfParams()).foreach { params =>
      System.setProperty("bigdl.engineType", "mkldnn")
      val conf = Engine.createSparkConf().setAppName("Test perf for convertion")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = params.batchSize
      val iterations = params.iteration

      val inputFormat = Memory.Format.nchw
      var input : Tensor[Float] = null
      var label : Tensor[Float] = null

      val modelLoad = Module.loadModule[Float](params.modelPath)
      val graph = if (!modelLoad.isInstanceOf[Graph[Float]]) modelLoad.toGraph() else modelLoad
      val model = graph.asInstanceOf[StaticGraph[Float]].toIRgraph(5, Memory.Format.nc)
      model.build()

      params.dataType match {
        case "imagenet" =>
          val inputShape = Array(batchSize, 3, 224, 224)
          input = Tensor(inputShape).rand()
          label = Tensor(batchSize).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 1000).toFloat)
        case "ssd" =>
          val inputShape = Array(batchSize, 3, 300, 300)
          input = Tensor(inputShape).rand()
          label = Tensor(batchSize).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 1000).toFloat)
        case _ => throw new UnsupportedOperationException(s"Unkown model ${params.dataType}")
      }

      model.evaluate()
      val broadcast = sc.broadcast(input, label)
      val length = batchSize * Engine.nodeNumber() * Engine.coreNumber() * iterations
      val rdd = sc.parallelize((1 to length), Engine.nodeNumber)
        .mapPartitions(iter => {
          val broad = broadcast.value
          Iterator.single(Sample(broad._1.clone().rand(), broad._2.clone().rand()))
        }).persist()

      val start = System.nanoTime()
      val result = model.evaluate(rdd,
        Array(new Top1Accuracy[Float]), Some(batchSize * Engine.nodeNumber() * Engine.coreNumber()))
      val takes = System.nanoTime() - start
      val avg = takes / iterations / 1e9
      logger.info(s"Average throughput is $avg imgs/sec")
    }
  }
}

case class DistriPerfParams (
  batchSize: Int = 4,
  iteration: Int = 50,
  dataType: String = "ssd",
  modelPath: String = "imagenet"
)
