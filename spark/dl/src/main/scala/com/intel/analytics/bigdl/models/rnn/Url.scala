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

package com.intel.analytics.bigdl.models.rnn

import breeze.linalg.*
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adagrad, Loss, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.model.Split

import scala.util.Random

object Url {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Train url detection on text")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init

    val batchSize = if (args.length > 0) {
      args(0).toInt
    } else {
      32 * 28 * 4
    }
    val totalLength = 13568
    var i = 0
    var model = if (args.length > 1) {
      if (args(1) == "cnn") {
        println("use buildCNN")
        buildCNN()
      } else if (args(1) == "rnn1") {
        println("use buildRNN 111")
        buildRNN()
      } else if (args(1) == "rnn2") {
        println("use buildRNN 222")
        buildRNN2()
      } else if (args(1) == "without") {
        buildWithout()
      } else if (args(1) == "linearRepeat") {
        println("linear repeat")
        val subModel = Sequential[Float]()
        val linear = TimeDistributed[Float](Linear[Float](20, 20))
        for (i <- 1 to 200) {
          subModel.add(linear)
        }
        subModel.add(Select[Float](2, -1))
        subModel.add(LogSoftMax[Float]())
        subModel
      } else if (args(1) == "lstm2") {
        println("use lstm 222")
        buildModel2()
      } else {
        println("use lstm 111")
        buildModel()
      }
    } else {
      println("use lstm 111")
      buildModel()
    }

    val inputSize = 36
    val times = if (args.length > 2) {
      args(2).toInt
    } else {
      200
    }

    val data = Array.tabulate(totalLength)(_ => Sample[Float]())
    val featureSize = Array(times, inputSize)
    val labelSize = Array(1)
    val label = Array(2.0f)
    while (i < totalLength) {
      val feature = Tensor[Float](times, inputSize).apply1(e => Random.nextFloat())
      data(i).set(feature.storage().array(), label, featureSize, labelSize)
      i += 1
    }

    val max_epoch = 20
    val state = T("learningRate" -> 0.01,
      "learningRateDecay" -> 0.0002)

    val trainSet = sc.parallelize(data, Engine.nodeNumber())
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainSet,
      criterion = ClassNLLCriterion[Float](),
      batchSize
    )

    optimizer.setState(state)
      .setOptimMethod(new Adagrad())
      .setEndWhen(Trigger.maxEpoch(max_epoch))
      .optimize()
  }

  def buildModel(class_num: Int = 2, vec_dim: Int = 36): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Recurrent[Float]()
      .add(LSTM[Float](vec_dim, 20)))
    .add(Select[Float](2, -1))
    .add(Linear[Float](20, class_num))
    .add(LogSoftMax[Float]())
    model
  }

  def buildModel2(class_num: Int = 2, vec_dim: Int = 20): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Recurrent[Float]()
      .add(LSTMNew[Float](vec_dim, 20)))
      .add(Select[Float](2, -1))
      .add(Linear[Float](20, class_num))
      .add(LogSoftMax[Float]())
    model
  }

  def buildRNN(class_num: Int = 2, vec_dim: Int = 36): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Recurrent[Float]()
      .add(RnnCell[Float](vec_dim, 20, Tanh())))
      .add(Select[Float](2, -1))
      .add(Linear[Float](20, class_num))
      .add(LogSoftMax[Float]())
    model
  }

  def buildRNN2(class_num: Int = 2, vec_dim: Int = 20): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Recurrent[Float]()
      .add(RnnCell2[Float](vec_dim, 20, Tanh())))
      .add(Select[Float](2, -1))
      .add(Linear[Float](20, class_num))
      .add(LogSoftMax[Float]())
    model
  }

  def buildWithout(class_num: Int = 2, vec_dim: Int = 20): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape[Float](Array(200*vec_dim)))
      .add(Linear[Float](200 * vec_dim, class_num))
      .add(LogSoftMax[Float]())
    model
  }

  def buildCNN(class_num: Int = 2): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape[Float](Array(20, 1, 200)))
    model.add(SpatialConvolution(20, 128, 5, 1))
    model.add(ReLU())
    model.add(SpatialMaxPooling(5, 1, 5, 1))
    model.add(SpatialConvolution(128, 128, 5, 1))
    model.add(ReLU())
    model.add(SpatialMaxPooling(5, 1, 5, 1))
    model.add(Reshape(Array(128 * 7)))
    model.add(Linear(128 * 7, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    model
  }
}
