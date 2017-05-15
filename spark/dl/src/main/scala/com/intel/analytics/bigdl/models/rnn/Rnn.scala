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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text.LabeledSentenceToSample
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import scala.util.Random

object Rnn {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Train rnn on text 2222")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init

    val batchSize = if (args.length > 0) {
      args(0).toInt
    } else {
      4 * 28 * 4
    }
    val totalLength = 27869 * 2

    val model = SimpleRNN(inputSize = 4001,
      hiddenSize = 40,
      outputSize = 4001)

    var i = 0
    val data = new Array[Sample[Float]](totalLength)
    var feature = Tensor[Float](22, 4001)
    var label = Tensor[Float](22)
    while (i < totalLength) {
      feature.apply1(e => Random.nextFloat())
      label.apply1(e => 1000.0f)
      data(i) = Sample(feature, label)
      i += 1
    }

    val trainSet = DataSet.array(data, sc).transform(SampleToBatch(batchSize))
    val state = T("learningRate" -> 0.1,
      "momentum" -> 0.0,
      "weightDecay" -> 0.0,
      "dampening" -> 0.0)

    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = TimeDistributedCriterion[Float](
        CrossEntropyCriterion[Float](), sizeAverage = true)
    )

    optimizer
      .setState(state)
      .setOptimMethod(new SGD())
      .setEndWhen(Trigger.maxEpoch(2))
      .optimize()
  }
}
