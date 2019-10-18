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

object Test {
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

      val map = MeanAveragePrecisionObjectDetection.createCOCO(81)

      val out = map(dt, gt)

      Engine.init
      val partitionNum = 2 // Engine.nodeNumber() * Engine.coreNumber()

      val url = "./coco-seq-0.seq"
      val rddData = DataSet.SeqFileFolder.filesToRoiImageFrame(url, sc, Some(partitionNum))
        .toDistributed().data(train = false)

      val batchSize = 1

      val transformer = MTImageFeatureToBatchWithResize(
          sizeDivisible = 32,
          batchSize = batchSize,
          transformer =
            PixelBytesToMat() ->
            ScaleResize(minSize = 800, maxSize = 1333) ->
            ChannelNormalize(122.7717f, 115.9465f, 102.9801f) ->
            MatToTensor[Float](),
            toRGB = false
        )
      val evaluationSet = transformer(rddData)

      val model = MaskTmpUtils.loadMaskModel() // Module.load[Float](param.model)

      val result = model.evaluate(evaluationSet,
        Array(MeanAveragePrecisionObjectDetection.createCOCO(81)))
      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    }}
  }
}
