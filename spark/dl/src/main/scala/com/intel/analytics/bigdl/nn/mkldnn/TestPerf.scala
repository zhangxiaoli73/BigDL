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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{DetectionOutputSSD, SpatialDilatedConvolution}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, LayerException, Util}
import org.apache.spark.SparkContext
import spire.syntax.module

object TestPerf {

  private def threadRun(module: Module[Float], input: Tensor[Float]): Unit = {
    val _subModelNumber = 4
    val workingModels = if (_subModelNumber != 1) {
      val wb = Util.getAndClearWeightBias(module.parameters())
      val models = (1 to _subModelNumber).map(i => {
        val m = module.cloneModule()
        Util.putWeightBias(wb, m)
        m.evaluate()
        m
      }).toArray
      Util.putWeightBias(wb, module)
      models
    } else {
      Array(module)
    }
    val s1 = System.nanoTime()
    println(Engine.wrapperComputing.getPoolSize)
    val trainingThreads = Engine.wrapperComputing.invoke((0 until _subModelNumber).map(i =>
      () => {
        try {
          val s1 = System.nanoTime()
          workingModels(i).forward(input)
          val end = (System.nanoTime() - s1) / 1e9
          println(s"blas_fwd_time ${end}")
        } catch {
          case e: Throwable =>
            val tmp = 0
            throw new LayerException(this.toString(), e)
        }
      }))
    Engine.default.sync(trainingThreads)

    val end = (System.nanoTime() - s1) / 1e9
    println(s"blas_threadRun_time ${end}")
  }

  private def run(module: Module[Float], input: Tensor[Float], part: Tensor[Float]): Unit = {
    println(s"batchSize ${input.size(1)}")

    for (i <- 0 to 10) {
      Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
        module.forward(input)
      }))
    }
    val s1 = System.nanoTime()
    for (i <- 0 to 9) {
      Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
        module.forward(input)
      }))
    }
    val end = (System.nanoTime() - s1) / 1e10
    println(s"blas_all_fwd_time_dnn ${end}")

    val s11 = System.nanoTime()
    for (i <- 0 to 9) {
      module.forward(input)
    }
    val end11 = (System.nanoTime() - s11) / 1e10
    println(s"blas_all_fwd_time ${end11}")

    val s2 = System.nanoTime()
    for (i <- 0 to 9) {
      module.forward(part)
    }
    val end2 = (System.nanoTime() - s2) / 1e10
    println(s"blas_all_part_time ${end2}")
  }


  def main(argv: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Test perf")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init

//    val module = SpatialDilatedConvolution[Float](512, 1024, 3, 3, 1, 1, 6, 6, 6, 6)
//    val batchSize = System.getProperty("batchSize", "8").toInt
//    val cores = Engine.coreNumber()
//    val input = Tensor[Float](batchSize, 512, 19, 19).rand()
//    val part = Tensor[Float](batchSize / cores, 512, 19, 19).rand()
//
//    run(module.asInstanceOf[Module[Float]], input, part)
//    threadRun(module.asInstanceOf[Module[Float]], part)

    val module = new DetectionOutputSSD[Float](21, true, 0, 0.45f, 400, 200, 0.0f, false, false)

    val in = Tensor[Float]
  }
}
