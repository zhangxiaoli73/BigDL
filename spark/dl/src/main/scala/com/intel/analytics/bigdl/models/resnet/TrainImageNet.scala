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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.mkldnn.ResNet_dnn
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object TrainImageNet {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils._

  def imageNetDecay(epoch: Int): Double = {
    if (epoch >= 80) {
      3
    } else if (epoch >= 60) {
      2
    } else if (epoch >= 30) {
      1
    } else {
      0.0
    }
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Train ResNet on ImageNet2012")
        // Will throw exception without this config when has only one executor
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = param.batchSize
      val (imageSize, dataSetType, maxEpoch, dataSet) =
        (224, DatasetType.ImageNet, 90, ImageNetDataSet)

      val trainDataSet = dataSet.trainDataSet(param.folder + "/train", sc, imageSize, batchSize)

      val validateSet = dataSet.valDataSet(param.folder + "/val", sc, imageSize, batchSize)

      val shortcut: ShortcutType = ShortcutType.B

      println("opt value " + param.optnet)

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else if (param.graphModel) {
        val curModel = ResNet_dnn(classNum = param.classes,
          T("shortcutType" -> ResNet_dnn.ShortcutType.B, "depth" -> param.depth,
            "optnet" -> param.optnet, "dataSet" -> ResNet_dnn.DatasetType.ImageNet))
//        val model = ResNet_dnn(classNum = 1000, T("depth" -> 50, "optnet" -> true,
//          "dataset" -> ResNet_dnn.DatasetType.ImageNet))
        ResNet_dnn.modelInit(curModel)
        curModel
      } else {
        val curModel =
          ResNet(classNum = param.classes, T("shortcutType" -> shortcut, "depth" -> param.depth,
          "optnet" -> param.optnet, "dataSet" -> dataSetType))
        if (param.optnet) {
          ResNet.shareGradInput(curModel)
        }
        ResNet.modelInit(curModel)
        curModel
      }

      println(model)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        val optim = OptimMethod.load[Float](param.stateSnapshot.get).asInstanceOf[LarsSGD[Float]]
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration
        optim.learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay)
        optim
      } else {
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration

        logger.info(s"warmUpIteraion: $warmUpIteration, startLr: ${param.learningRate}, " +
          s"maxLr: $maxLr, " +
          s"delta: $delta, nesterov: ${param.nesterov}")
//        new LarsSGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
//          momentum = param.momentum,
////          larsLearningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay),
//          larsLearningRateSchedule =
//            SGD.PolyWithWarmUp(warmUpIteration, delta, 2, iterationsPerEpoch * 90),
//          gwRation = 0.001
//        )
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
        momentum = param.momentum, dampening = param.dampening,
        nesterov = param.nesterov,
        learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay))
//        learningRateSchedule = SGD.PolyWithWarmUp(warmUpIteration, delta, 2, iterationsPerEpoch * 90))
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new CrossEntropyCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      
      val logdir = "resnet-imagenet"
      val appName = s"${sc.applicationId}"
      val trainSummary = TrainSummary(logdir, appName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
      val validationSummary = ValidationSummary(logdir, appName)

      optimizer
        .setOptimMethod(optimMethod)
//        .setTrainSummary(trainSummary)
//        .setValidationSummary(validationSummary)
        .setValidation(Trigger.everyEpoch,
          validateSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
      sc.stop()

    })
  }
}
