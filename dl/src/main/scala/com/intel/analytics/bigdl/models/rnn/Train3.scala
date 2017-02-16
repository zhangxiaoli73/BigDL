/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.rnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text.{LabeledSentence, LabeledSentenceToSample, SampleToBatchPadding}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext

object Train3{

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val node = 1
      val core = 1
      val scc = Engine.init(node, core, true).map(conf => {
        conf.setAppName("Train Lenet on MNIST")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
          .setMaster("local[1]")
        new SparkContext(conf)
      })
      val sc = scc.get

      val totalSize = 80
      val trainData = new Array[LabeledSentence[Float]](totalSize)
      var i = 0
      var base = Array(1.0f)
      while (i < totalSize) {
        val input = Array(0.0f) ++ base ++ Array(100.0f)
        val label = Array(1.0f) ++ base ++ Array(100.0f)
        trainData(i) = new LabeledSentence[Float](input, label)
        i += 1
        base = base ++ Array(i.toFloat)
      }

      val trainMaxLength = 20
      val batchSize = 4
      val dictionaryLength = 4001
      val trainSet = DataSet.array(trainData, sc)
            .transform(LabeledSentenceToSample(dictionaryLength))
            .transform(SampleToBatchPadding[Float](batchSize, true))

      /*
      val data = trainSet.toDistributed().data(train = true)
      val tmp = data.mapPartitions(iter => {
        iter.map(batch => {
          val input = batch.data
          val label = batch.labels
          label.nElement() / batchSize
        })
      })

      val tmp1 = tmp.collect()
      sys.exit(1)
      */

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = SimpleRNN(
          inputSize = dictionaryLength,
          hiddenSize = param.hiddenSize,
          outputSize = dictionaryLength,
          bpttTruncate = param.bptt)
        curModel.reset()
        curModel
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T("learningRate" -> param.learningRate,
          "momentum" -> param.momentum,
          "weightDecay" -> param.weightDecay,
          "dampening" -> param.dampening)
      }

      Engine.init(1, param.coreNumber, false)
      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new CrossEntropyCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      optimizer
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()
    })
  }
}
