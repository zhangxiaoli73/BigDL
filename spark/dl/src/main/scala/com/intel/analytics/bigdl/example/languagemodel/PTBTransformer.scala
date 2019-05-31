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

package com.intel.analytics.bigdl.example.languagemodel

import java.io.File

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.text.{Dictionary, LabeledSentenceToSample, _}
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.example.languagemodel.Utils._
import com.intel.analytics.bigdl.models.rnn.SequencePreprocess
import org.dmg.pmml.False

object PTBTransformer {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val conf = Engine.createSparkConf()
        .setAppName("Train ptbModel on text")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val (trainData, validData, dictionary) = SequencePreprocess.transformer(
        param.dataFolder, param.vocabSize)

      val maxLength = 100
      val trainSet = DataSet.rdd(sc.parallelize(trainData))
        .transform(TextToLabeledSentence[Float]())
        .transform(LabeledSentenceToSample[Float](
          oneHot = false,
          fixDataLength = Some(maxLength),
          fixLabelLength = Some(maxLength)))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      val validationSet = DataSet.rdd(sc.parallelize(validData))
        .transform(TextToLabeledSentence[Float]())
        .transform(LabeledSentenceToSample[Float](
          oneHot = false,
          fixDataLength = Some(maxLength),
          fixLabelLength = Some(maxLength)))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      val model = if (param.modelSnapshot.isDefined) {
        Module.loadModule[Float](param.modelSnapshot.get)
      } else {
        val curModel = PTBModel.transformer(
          inputSize = param.vocabSize,
          hiddenSize = param.hiddenSize,
          outputSize = param.vocabSize,
          numLayers = param.numLayers)
        curModel.reset()
        curModel
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        // ('learning_rate', 0.2), ('learning_rate_constant', 2.0),
        // ('learning_rate_cosine_cycle_steps', 250000), ('learning_rate_decay_rate', 1.0),
        // ('learning_rate_decay_scheme', 'noam'), ('learning_rate_decay_staircase', False), ('learning_rate_decay_steps', 5000), ('learning_rate_minimum', None), ('learning_rate_schedule', 'constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size'), ('learning_rate_warmup_steps', 8000),
        new Adam[Float](learningRate = param.learningRate,
          learningRateDecay = param.learningRateDecay, beta2 = 0.997)
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = TimeDistributedCriterion[Float](
          CrossEntropyCriterion[Float](), sizeAverage = false, dimension = 2)
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
            sizeAverage = false, dimension = 2))))
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()
      sc.stop()
    })
  }
}
