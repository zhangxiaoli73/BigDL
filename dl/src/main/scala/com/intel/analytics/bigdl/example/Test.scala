/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.example

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, MiniBatch}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim.{Optimizer, Top1Accuracy, Trigger, Validator}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.{DataSet, Module}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object BigDLSample {
  Logger.getLogger("org").setLevel(Level.INFO)
  Logger.getLogger("breeze")

  val trainDataFile = "train-images-idx3-ubyte"
  val trainLabelFile = "train-labels-idx1-ubyte"
  val validationDataFile = "t10k-images-idx3-ubyte"
  val validationLabelFile = "t10k-labels-idx1-ubyte"

  private var nodeNumber = 1
  private var coreNumber = 4
  private var mult = 64

  lazy val batchSize = {
    nodeNumber * coreNumber * mult
  }

  val trainData = s"/data/mnist/$trainDataFile"
  val trainLabel = s"/data/mnist/$trainLabelFile"
  val validationData = s"/data/mnist/$validationDataFile"
  val validationLabel = s"/data/mnist/$validationLabelFile"

  val trainMean = 0.13066047740239506
  val trainStd = 0.3081078

  val testMean = 0.13251460696903547
  val testStd = 0.31048024


  var sc : SparkContext = null

  def execute(): Unit = {
    println(s"nodeNumber: $nodeNumber coreNumber: $coreNumber mult: $mult")
    // Engine.init(nodeNumber, coreNumber, true /* env == "spark" */)
	  val scc = Engine.init(nodeNumber, coreNumber, true).map(conf => {
        conf.setAppName(this.getClass.getSimpleName)
        new SparkContext(conf)
     })
	  sc = scc.get
    val trainSet = DataSet.array(load(trainData, trainLabel), sc) ->
      BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(batchSize)

    val validationSet = DataSet.array(load(validationData, validationLabel), sc) ->
      BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(batchSize)

    val model = train(trainSet, validationSet)
    test(model, validationSet)
  }


  def test(trainedModel: Module[Float],
           validationSet: DataSet[MiniBatch[Float]]): Unit = {
    val validator = Validator(trainedModel, validationSet)
    val result = validator.test(Array(new Top1Accuracy[Float]))
    result.foreach(r => {
      println(s"${r._2} is ${r._1}")
    })
  }


  def train(trainSet: DataSet[MiniBatch[Float]],
            validationSet: DataSet[MiniBatch[Float]]): Module[Float] = {
    val state = T("learningRate" -> 0.05 / 4 * mult)
    val maxEpoch = 15

    val initialModel = LeNet5(10)  // 10 digit classes
    // val initialModel = ModelBuilder(10) // vgg model
    val optimizer = Optimizer(
      model = initialModel,
      dataset = trainSet,
      criterion = ClassNLLCriterion[Float]())
    val trainedModel = optimizer
      .setValidation(
        trigger = Trigger.everyEpoch,
        dataset = validationSet,
        vMethods = Array(new Top1Accuracy))
      .setState(state)
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .optimize()

    trainedModel
  }


  def loadBinaryFile(filePath: String): Array[Byte] = {
    val files = sc.binaryFiles(filePath)
    val bytes = files.first()._2.toArray()
    bytes
  }


  def load(featureFile: String, labelFile: String): Array[ByteRecord] = {
    val labelBuffer = ByteBuffer.wrap(loadBinaryFile(labelFile))
    val featureBuffer = ByteBuffer.wrap(loadBinaryFile(featureFile))
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }
    result
  }


  def main(args: Array[String]): Unit = {
    for (i <- 0 until args.length) {
      val a = args(i)
      if (a == "--node_number") {
        nodeNumber = args(i + 1).toInt
      }
      if (a == "--core_number") {
        coreNumber = args(i + 1).toInt
      }
      if (a == "--mult") {
        mult = args(i + 1).toInt
      }
    }
    var t = System.currentTimeMillis()
    execute()
    t = System.currentTimeMillis() - t
    println(s"process time: $t [ms]")
  }

}
