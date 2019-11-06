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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

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

//    val modelPath = "/home/zhangli/CodeSpace/forTrain/coco-2017/maskrcnn-state.model"
//    mask.saveModule(modelPath)
//    Module.loadModule[Float](modelPath)
  }
}

object GenerateModel {
  def main(args: Array[String]): Unit = {
   val m = MaskTmpUtils.loadMaskModel()

    println("done")
  }
}
