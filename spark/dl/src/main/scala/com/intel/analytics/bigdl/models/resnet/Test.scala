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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
import com.intel.analytics.bigdl.models.resnet.Utils._
import com.intel.analytics.bigdl.optim.{Top1Accuracy, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToSample, BytesToBGRImg, SampleToImagefeature}
import com.intel.analytics.bigdl.models.resnet.ResNet.ShortcutType
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import scala.collection.mutable.ArrayBuffer

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("Test ResNet on Cifar10")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init
      val partitionNum = Engine.nodeNumber() * Engine.coreNumber()

      val rddData = sc.parallelize(loadTest(param.folder), partitionNum)
      val transformer = BytesToBGRImg() -> BGRImgNormalizer(Cifar10DataSet.trainMean,
          Cifar10DataSet.trainStd) -> BGRImgToSample() -> SampleToImagefeature()
      val evaluationSet = transformer(rddData)

      RandomGenerator.RNG.setSeed(10)
      val model = ResNet(10, T("shortcutType" -> ShortcutType.A, "depth" -> 20, "optnet" -> false))
      // Module.load[Float](param.model)
      // val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float]),
      //  Some(param.batchSize))
      // result.foreach(r => println(s"${r._2} is ${r._1}"))
      val result = model.predictImage(ImageFrame.rdd(evaluationSet), batchPerPartition = 2) // . toDistributed().rdd
      val tmpRes = result.toDistributed().rdd.mapPartitions(res => {
        res.map(feature => {
         val label = feature.apply[Tensor[Float]]("label")
         val predict = feature.apply[Tensor[Float]]("predict").max(1)._2
        println(label.valueAt(1) + "*******************" + predict.valueAt(1))
          if (label.valueAt(1) == predict.valueAt(1)) {
            1
          } else {
            0
          }
        })
      }).reduce((left, right) => {
        left + right
      })
      println(s"accuracy ${tmpRes.toDouble / evaluationSet.count()} right $tmpRes total ${evaluationSet.count()}")
      sc.stop()
    }
  }
}
