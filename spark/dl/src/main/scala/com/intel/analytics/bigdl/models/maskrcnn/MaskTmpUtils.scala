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

import java.io.File

import breeze.numerics._
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelNormalize, ScaleResize}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.spark.SparkContext
import org.apache.commons.io.FileUtils

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object MaskTmpUtils {
  def loadWeight(path: String, size: Array[Int]): Tensor[Float] = {
    val weight = Tensor[Float](size)
    val arr = weight.storage().array()

    var totalLength = 0
    for (line <- Source.fromFile(path).getLines) {
      val ss = line.split(",")
      for (i <- 0 to ss.length - 1) {
        if (ss(i) != "") {
          arr(totalLength) = ss(i).toFloat
          totalLength += 1
        }
      }
    }

    require(totalLength == size.product, s"total length ${totalLength} ${size.product}")
    weight
  }

  def loadMaskModel(): Module[Float] = {
    val resNetOutChannels = 256
    val backboneOutChannels = 256
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels, numClasses = 81)

    val params = mask.getParametersTable()
    val keys = params.keySet
    val path = "/home/zhangli/workspace/framework/detectron2/weights-dir/"
    for(i <- keys) {
      // for weight
      var p = params.get[Table](i).get.get[Tensor[Float]]("weight").get
      var size = p.size()
      var name = path + i.toString + ".weight"
      if (size(0) != 1) {
        size.foreach(n => name = name + s"_${n}")
      } else {
        size.slice(1, size.length).foreach(n => name = name + s"_${n}")
      }

      name = name + ".txt"
      val weight = MaskTmpUtils.loadWeight(name, size)
      p.set(weight)

      // for bias
      p = params.get[Table](i).get.get[Tensor[Float]]("bias").getOrElse(null)
      if (p != null) {
        size = p.size()
        name = path + i.toString + ".bias"
        size.foreach(n => name = name + s"_${n}")
        name = name + ".txt"
        val bias = MaskTmpUtils.loadWeight(name, size)
        p.set(bias)
      }

      // for running mean
      p = params.get[Table](i).get.get[Tensor[Float]]("runningMean").getOrElse(null)
      if (p != null) {
        size = p.size()
        name = path + i.toString + ".running_mean"
        size.foreach(n => name = name + s"_${n}")
        name = name + ".txt"
        val bias = MaskTmpUtils.loadWeight(name, size)
        p.set(bias)
      }

      // for running variance
      p = params.get[Table](i).get.get[Tensor[Float]]("runningVar").getOrElse(null)
      if (p != null) {
        size = p.size()
        name = path + i.toString + ".running_var"
        size.foreach(n => name = name + s"_${n}")
        name = name + ".txt"
        val bias = MaskTmpUtils.loadWeight(name, size)
        p.set(bias)
      }
      println(s"${i} done")
    }
    mask

//    val modelPath = "/home/zhangli/CodeSpace/Datasets/coco-2017/mask-models/maskrcnn-detectron2.model"
//    mask.saveModule(modelPath, overWrite = true)
//    Module.loadModule[Float](modelPath)
  }
}

object GenerateModel {

  def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
    val absA = math.abs(a)
    val absB = math.abs(b)
    val diff = math.abs(a - b)

    val result = if (a == b) {
      true
    } else {
      math.min(diff / (absA + absB), diff) < epsilon
    }

    result
  }

  def nearequals(t1: Tensor[Float], t2: Tensor[Float],
                 epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    result
  }

  def main(args: Array[String]): Unit = {
//    val conf = Engine.createSparkConf().setAppName("Test MaskRCNN on COCO")
//      .set("spark.akka.frameSize", 64.toString)
//      .set("spark.task.maxFailures", "1")
//    val sc = new SparkContext(conf)
//    Engine.init

//    val imagePath = "/home/zhangli/workspace/framework/detectron2/datasets/coco/val2017/000000000139.jpg"
//
//    val bytes = FileUtils.readFileToByteArray(new File(imagePath))
//    val imagefeature = ImageFeature(bytes)
//
//    // transformer
//    val trans1 = BytesToMat() -> ScaleResize(minSize = 800, maxSize = 1333) -> MatToTensor[Float](toRGB = false)
//
//    // load as BGR,  then convert to HWC
//
//    val out1 = trans1.transform(imagefeature)
//
//    val image = out1[Tensor[Float]]("imageTensor").transpose(1, 2).transpose(2, 3)
//    val inputPytorch = MaskTmpUtils.loadWeight("/home/zhangli/workspace/dummy_input.txt", Array(1, 3, 800, 1216))
//
//    val trans = BytesToMat() -> ScaleResize(minSize = 800, maxSize = 1333) ->
//      ChannelNormalize(123.6750f, 116.2800f, 103.5300f) -> MatToTensor[Float]()
//
//    val outFeature = trans.transform(imagefeature)
//
//    println("done")
//
//
//    val rddData = DataSet.SeqFileFolder.filesToRoiImageFeatures("./coco-seq-0.seq",
//      sc, Some(1)).toDistributed().data(train = false)
//
//    val transformer = MTImageFeatureToBatchWithResize(
//      sizeDivisible = 32,
//      batchSize = 1,
//      transformer =
//        PixelBytesToMat() ->
//          ScaleResize(minSize = 800, maxSize = 1333) ->
//          ChannelNormalize(123.6750f, 116.2800f, 103.5300f) ->
//          MatToTensor[Float](),
//      toRGB = false
//    )
//    val evaluationSet = transformer(rddData).collect()

//    val resNetOutChannels = 256
//    val backboneOutChannels = 256
//    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels, numClasses = 81)
//    val res = mask // .buildResNet50()
//
//    val table = res.getParameters()
//    table._1.fill(0.0f)
//    table._2.fill(0.0f)
//
//    val extra = res.getExtraParameter()
//    for (i <- 0 until extra.length) {
//      extra(i).fill(0.1f)
//    }
//
//    res.saveModule("/tmp/resnet.model", overWrite = true)
//    val resLoad = Module.loadModule[Float]("/tmp/resnet.model")
//
//    val pp = resLoad.getExtraParameter()
//    val pp2 = res.getExtraParameter()
//
//    println("done")




    val m = MaskTmpUtils.loadMaskModel()
    val modelPath = "/home/zhangli/CodeSpace/Datasets/coco-2017/mask-models/maskrcnn-detectron2.model"
    val m_load = Module.loadModule[Float](modelPath)

    val e1 = m.getExtraParameter()
    val e2 = m_load.getExtraParameter()

    val p1 = m.getParametersTable()
    val p2 = m_load.getParametersTable()

    val keys = p1.keySet
    val path = "/home/zhangli/workspace/framework/detectron2/weights-dir/"
    for(i <- keys) {
      // for weight
      var p11 = p1.get[Table](i).get.get[Tensor[Float]]("weight").get
      var p22 = p2.get[Table](i).get.get[Tensor[Float]]("weight").get

      require(nearequals(p11, p22))

      // for bias
      p11 = p1.get[Table](i).get.get[Tensor[Float]]("weight").get
      p22 = p2.get[Table](i).get.get[Tensor[Float]]("weight").get

      require(nearequals(p11, p22))

      // for running mean
      p11 = p1.get[Table](i).get.get[Tensor[Float]]("runningMean").getOrElse(null)
      p22 = p2.get[Table](i).get.get[Tensor[Float]]("runningMean").getOrElse(null)
      if (p11 != null && p22 != null) require(nearequals(p11, p22))

      // for running variance
      p11 = p1.get[Table](i).get.get[Tensor[Float]]("runningVar").getOrElse(null)
      p22 = p2.get[Table](i).get.get[Tensor[Float]]("runningVar").getOrElse(null)
      if (p11 != null && p22 != null) require(nearequals(p11, p22))
    }

    println("tests done")
    val input = MaskTmpUtils.loadWeight("/home/zhangli/workspace/dummy_input.txt", Array(1, 3, 800, 1216))

    m.evaluate()
    val out = m.forward(T(input, Tensor[Float](T(T(800.0f, 1202.0f, 426.0f, 640.0f)))))

    m_load.evaluate()
    val out_load = m_load.forward(T(input, Tensor[Float](T(T(800.0f, 1202.0f, 426.0f, 640.0f)))))

    println("done")
  }
}
