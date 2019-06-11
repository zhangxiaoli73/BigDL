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
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.CropCenter
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, StaticGraph}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, MTImageFeatureToBatch, MatToTensor, PixelBytesToMat}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelScaledNormalizer, RandomCropper, RandomResize}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.intermediate.ConversionUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
 * This example is to evaluate trained resnet50 with imagenet data and get top1 and top5 accuracy
 */
object TestImageNet {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Test model on ImageNet2012")
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val modelCaffe = Module.loadModule[Float]("/home/zhangli/workspace/zoo-model/analytics-zoo_resnet-50_imagenet_0.1.0.model")

      val batch = Tensor[Float](64, 3, 224, 224).apply1(_ =>
        RandomGenerator.RNG.uniform(-100, 100).toInt.toFloat)
      val inputTemp = batch.select(1, 1).resize(Array(1, 3, 224, 224))
      val model2 = modelCaffe.cloneModule()

      modelCaffe.evaluate()
      model2.evaluate()

      val out1 = modelCaffe.forward(batch)
      val out2 = model2.forward(inputTemp).toTensor[Float].resize(Array(1, 1000))

      println("done")

      import com.intel.analytics.bigdl.models.resnet
      val model = ResNet.graph(classNum = 1000, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> false, "dataset" -> DatasetType.ImageNet))
      // Module.loadModule[Float](param.model)
      val modelDnn = ConversionUtils.convert(model.cloneModule())

      val evaluationSet = ImageNetDataSet.valDataSet(param.folder,
        sc, 224, param.batchSize).toDistributed().data(train = false)

      val criterion = new CrossEntropyCriterion[Float]()
      val criterionDnn = new CrossEntropyCriterion[Float]()

      val data = evaluationSet.mapPartitions(batch => {
        batch.map(dd => {
          (dd.getInput().toTensor[Float].clone(), dd.getTarget().toTensor[Float].clone())
        })
      }).collect()

      val length = data.length
      var i = 0
      while (i < length) {
        val input = data(i)._1
        val target = data(i)._2

        val out1 = model.forward(input).toTensor[Float]
        val out2 = modelDnn.forward(input)

        criterion.forward(out1, target)
        val cri = criterion.backward(out1, target)

        val grad1 = model.backward(input, cri)
        val grad2 = modelDnn.backward(input, cri)

        model.zeroGradParameters()
        modelDnn.zeroGradParameters()

        i += 1
      }
//      val result = model.evaluate(evaluationSet,
//        Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
//      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    })
  }
}
