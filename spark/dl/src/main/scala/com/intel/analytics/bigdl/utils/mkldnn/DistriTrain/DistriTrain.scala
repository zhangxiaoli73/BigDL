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
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Graph, Module, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
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

object DistriTrain {

  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[DistriTrainParams]("BigDL w/ Dnn Local Model Performance Test") {
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

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriTrainParams()).foreach { params =>
      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = params.batchSize
      val iterations = params.iteration

      val modelLoad = if (params.modelPath == "lenet") {
        LeNet5.graph(10)
      } else if (params.modelPath == "inceptionV1") {
        Inception_v1_NoAuxClassifier.graph(1000)
      } else {
        Module.loadModule[Float](params.modelPath)
      }

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
          inputShape = Array(batchSize, 3, 224, 224)
          outputShape = Array(batchSize)
        case "ssd" =>
          inputShape = Array(batchSize, 3, 300, 300)
          outputShape = Array(batchSize)
        case "lenet" =>
          inputShape = Array(batchSize, 1, 28, 28)
          outputShape = Array(batchSize)
          case _ => throw new UnsupportedOperationException(s"Unkown model ${params.dataType}")
      }

      val broadcast = sc.broadcast(inputShape, outputShape)
      val length = batchSize * iterations
      val rdd = sc.parallelize((1 to length), Engine.nodeNumber)
        .mapPartitions(iter => {
          val broad = broadcast.value
          val inputShape = broad._1
          val outputShape = broad._2
          val t = new ArrayBuffer[MiniBatch[Float]]()
          while (iter.hasNext) {
            iter.next()
            val input = Tensor(inputShape).rand()
            val label = Tensor(outputShape).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 10).toFloat)
            t.append(MiniBatch(input, label))
          }
          t.toIterator
        }).persist()

      val optimizer = Optimizer(
        model, DataSet.rdd(rdd), CrossEntropyCriterion[Float]())

      optimizer.setOptimMethod(new SGD())
        .setEndWhen(Trigger.maxIteration(params.iteration))
        .optimize()
      sc.stop()
    }
  }
}

case class DistriTrainParams (
    batchSize: Int = 4,
    iteration: Int = 80,
    dataType: String = "lenet",
    modelPath: String = "imagenet",
    modelType: String = "mkldnn"
  )
