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

    val totalLength = 6784
    val node = 4
    var i = 0
    val data = Array.tabulate(totalLength)(_ => Sample[Float]())
    val featureSize = Array(200, 36)
    val labelSize = Array(1)
    val label = Array(2.0f)
    while (i < totalLength) {
      val feature = Tensor[Float](200, 36).apply1(e => Random.nextFloat())
      data(i).set(feature.storage().array(), label, featureSize, labelSize)
      i += 1
    }

    val training_split = 0.8
    val batchSize = 32 * 28 * node
    val max_epoch = 20
    val state = T("learningRate" -> 0.01,
      "learningRateDecay" -> 0.0002)

    val trainSet = sc.parallelize(data)

    val model = buildModel()
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
      .add(LSTM[Float](36, 20)))
    .add(Select[Float](2, -1))
    .add(Linear[Float](20, class_num))
    .add(LogSoftMax[Float]())
    model
  }
}
