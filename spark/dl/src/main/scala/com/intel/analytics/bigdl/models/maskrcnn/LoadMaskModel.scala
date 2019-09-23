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

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, Table}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object LoadMaskModel {
  def main(args: Array[String]): Unit = {
    RandomGenerator.RNG.setSeed(100)
    val resNetOutChannels = 256
    val backboneOutChannels = 256
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels)

    val params = mask.getParametersTable()
    val keys = params.keySet
    val path = "/home/zhangli/workspace/tmp/mask/maskrcnn-benchmark/demo/weight/"
//    for(i <- keys) {
//      // for weight
//      var p = params.get[Table](i).get.get[Tensor[Float]]("weight").get
//      var size = p.size()
//      var name = path + i.toString + ".weight"
//      if (size(0) != 1) {
//        size.foreach(n => name = name + s"_${n}")
//      } else {
//        size.slice(1, size.length).foreach(n => name = name + s"_${n}")
//      }
//
//      name = name + ".txt"
//      val weight = MaskUtils.loadWeight(name, size)
//      p.set(weight)
//
//      // for bias
//      p = params.get[Table](i).get.get[Tensor[Float]]("bias").getOrElse(null)
//      if (p != null) {
//        size = p.size()
//        name = path + i.toString + ".bias"
//        size.foreach(n => name = name + s"_${n}")
//        name = name + ".txt"
//        val bias = MaskUtils.loadWeight(name, size)
//        p.set(bias)
//      }
//
//      println(s"${i} done")
//    }

    val input = MaskUtils.loadWeight(path + "input.txt", Array(1, 3, 800, 1088))
    val inputClone = input.clone()

    mask.evaluate()
    val out = mask.forward(input).toTable

//    mask.save("/home/zhangli/workspace/tmp/mask/maskrcnn-benchmark/demo/mask.model", overWrite = true)

  val model = Module.load[Float](
    "/home/zhangli/workspace/tmp/mask/maskrcnn-benchmark/demo/mask.model")

    model.evaluate()
    val out2 = model.forward(inputClone)
    println("done")
  }
}
