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

package com.intel.analytics.bigdl.models.maskrcnn

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder
import com.intel.analytics.bigdl.dataset.image.CropRandom
import com.intel.analytics.bigdl.models.resnet.Utils.{TestParams, _}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.utils.{Engine, T}
import scopt.OptionParser
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, segmentation}
import com.intel.analytics.bigdl.dataset.segmentation.COCODataset
import com.intel.analytics.bigdl.models.mask.CoCo
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.intermediate.ConversionUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

object TestDebug {
  case class TestParams(
                         folder: String = "./",
                         model: String = "",
                         batchSize: Int = 128
                       )

  val testParser = new OptionParser[TestParams]("BigDL ResNet on Cifar10 Test Example") {
    opt[String]('f', "folder")
      .text("the location of COCO dataset")
      .action((x, c) => c.copy(folder = x))

    opt[String]('m', "model")
      .text("the location of model snapshot")
      .action((x, c) => c.copy(model = x))

    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

  def testMiniBatch(dataset: RDD[MiniBatch[Float]],
                    model : Module[Float]): Unit = {
    val rdd = dataset
    val modelBroad = ModelBroadcast[Float]().broadcast(rdd.sparkContext, model.evaluate())

    val out = rdd.mapPartitions(miniBatch => {
      val localModel = modelBroad.value()
      miniBatch.map(batch => {
        val input = batch.getInput()
        val target = null // batch.getTarget()

        val output = localModel.forward(input)
        (output, target)
      })
    }).collect()
  }

  def loadFromSource(): Array[ImageFeature] = {
    val f = "/home/zhangli/CodeSpace/forTrain/coco-2017/val2017"
    val m = "/home/zhangli/CodeSpace/forTrain/coco-2017/annotations/instances_val2017.json"

    val tt = CoCo.loadDetectionBBox(m)
    println("done")
    //    val fileNames = Array[String](
    //      "000000000139.jpg", "000000000285.jpg", "000000000632.jpg",
    //      "000000000724.jpg", "000000000776.jpg", "000000000785.jpg",
    //      "000000000802.jpg", "000000000872.jpg", "000000000885.jpg")

    val fileNames = Array[String](
      "000000000632.jpg")


    val features = new ArrayBuffer[ImageFeature]()
    val meta = COCODataset.load(m)
    meta.images.foreach(img => {
      if (fileNames.contains(img.fileName)) {
        val bytes = Files.readAllBytes(Paths.get(f, img.fileName))
        val feature = ImageFeature(bytes)
        features.append(ImageFeature(bytes))
        println("done")
      }
    })
    features.toArray
  }

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach { param => {
      val conf = Engine.createSparkConf().setAppName("Test MaskRCNN on COCO")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      val f = "/home/zhangli/CodeSpace/forTrain/coco-2017/val2017"
      val m = "/home/zhangli/CodeSpace/forTrain/coco-2017/annotations/instances_val2017.json"

      val p = "/home/zhangli/workspace/tmp/mask/maskrcnn-benchmark/tools/inference/coco_2014_minival/bbox.json"

      val dt = CoCo.loadDetectionBBox(p)
      val gt = CoCo.loadDetectionBBox("/home/zhangli/workspace/tmp/mask/maskrcnn-benchmark/tools/inference/coco_2014_minival/bbox-gt.json")

      // val map = MeanAveragePrecisionObjectDetection.createCOCO(81)

      //      val out = map(dt, gt)

      Engine.init
      val partitionNum = 2 // Engine.nodeNumber() * Engine.coreNumber()
      val rddData = DataSet.array[ImageFeature](loadFromSource(), sc).toDistributed().data(train = false)

      val minSize = 800
      val maxSize = 1333
      val batchSize = 1

      val transformer = MTImageFeatureToBatchWithResize(
        sizeDivisible = 32,
        batchSize = batchSize,
        transformer = BytesToMat() ->
          ScaleResize(minSize, maxSize, resizeROI = true) ->
          ChannelNormalize(122.7717f, 115.9465f, 102.9801f) ->
          MatToTensor[Float](),
        toRGB = false
      )

      val evaluationSet = transformer.apply(rddData)
      val model = MaskTmpUtils.loadMaskModel()

      println("done")
      sc.stop()
    }}
  }
}
