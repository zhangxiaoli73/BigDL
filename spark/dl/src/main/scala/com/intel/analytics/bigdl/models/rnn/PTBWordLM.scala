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
import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, PaddingParam, SampleToMiniBatch}
import com.intel.analytics.bigdl.dataset.text.LabeledSentenceToSample
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim.SGD.EpochLearningDecay
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object PTBWordLM {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val conf = Engine.createSparkConf()
        .setAppName("Train rnn on text")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val (trainData, validData, testData, dictionary) = SequencePreprocess(
        param.dataFolder, param.vocabSize)

      val trainSet = DataSet.rdd(sc.parallelize(
        SequencePreprocess.reader(trainData, param.numSteps)))
          .transform(TextToLabeledSentence[Float](param.numSteps))
          .transform(LabeledSentenceToSample[Float](
            oneHot = false,
            fixDataLength = None,
            fixLabelLength = None))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      val validationSet = DataSet.rdd(sc.parallelize(
        SequencePreprocess.reader(validData, param.numSteps)))
        .transform(TextToLabeledSentence[Float](param.numSteps))
        .transform(LabeledSentenceToSample[Float](
          oneHot = false,
          fixDataLength = None,
          fixLabelLength = None))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        println(s"inter number ${param.interNum}")
        val curModel = PTBModel(
          inputSize = param.vocabSize,
          hiddenSize = param.hiddenSize,
          outputSize = param.vocabSize,
          numLayers = param.numLayers,
          interNum = param.interNum,
          keepProb = param.keepProb)
        curModel.reset()
        curModel
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate, clipNorm = param.clipNorm,
          learningRateDecay = param.learningRateDecay,
          learningRateSchedule = EpochLearningDecay(param.startEpoch, param.nEpochs))
      }

      val logdir = "ptbModel"
      val appName = s"${sc.applicationId}"
      val trainSummary = TrainSummary(logdir, appName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
      val validationSummary = ValidationSummary(logdir, appName)

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = TimeDistributedCriterion[Float](
          CrossEntropyCriterion[Float](), sizeAverage = false)
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      if(param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float](
          TimeDistributedCriterion[Float](
            CrossEntropyCriterion[Float](),
            sizeAverage = false))))
        .setTrainSummary(trainSummary)
        .setValidationSummary(validationSummary)
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()
      sc.stop()
    })
  }
}
