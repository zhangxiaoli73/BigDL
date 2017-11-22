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

package com.intel.analytics.bigdl.example

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

object DeepSpeechPerf {

  var warmup = 10
  var iterations = 5

  def main(args: Array[String]): Unit = {
    val modelPath = "/home/zhangli/CodeSpace/Sprint31/DeepSpeech/model" // args(0)
    val uttLen = 3000
    val batchSize = 1
    val input = Tensor[Float](batchSize, 1, 13, uttLen).rand()
    val labels = Tensor[Float](batchSize, uttLen / 3).fill(1.0f)


    val conf = Engine.createSparkConf()
      .setAppName("test deepspeech 2 on text")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init

    val model = DeepSpeech2ModelLoader.loadModel(sc, modelPath)

    println(model)
    run(model, input)
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)], allSum: Long): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long, Long, Double)]
    var i = 0
    while (i < times.length) {
      val rate = times(i)._2.toDouble/ allSum
      timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, rate))
      i += 1
    }
    val sortData = timeBuffer.sortBy(a => a._4)
    println("111111111111111111        ")
    sortData.foreach(println)
  }

  def run(model: Module[Float], input: Tensor[Float]): Unit = {
    println("start warm up")
    for (i <- 0 to warmup) {
      println(i)
      model.forward(input)
    }

    println("start run iterations")
    val s1 = System.nanoTime()
    for (i <- 0 to iterations) {
      val s1 = System.nanoTime()
      model.forward(input)
      val end1 = System.nanoTime() - s1
      val tmp = model.getTimes()
      getTopTimes(tmp, end1)
      model.resetTimes()
    }
//    for (i <- 0 to iterations) {
//      model.forward(input)
//    }
    val end1 = System.nanoTime() - s1
    println("time " + end1/1e9 + " s")
  }

}
