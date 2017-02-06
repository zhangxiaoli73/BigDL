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

import java.util

import com.intel.analytics.bigdl.dataset.{SampleToBatch, Transformer, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.{DataSet => _, _}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
// import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object ToSample {
  def apply(nRows: Int, nCols: Int)
  : ToSample =
    new ToSample(nRows, nCols)
}

class ToSample(nRows: Int, nCols: Int)
  extends Transformer[(Array[Double], Double), Sample[Double]] {

  private val buffer = new Sample[Double]()
  private var featureBuffer: Array[Double] = null
  private var labelBuffer: Array[Double] = null

  override def apply(prev: Iterator[(Array[Double], Double)]): Iterator[Sample[Double]] = {

    prev.map(x => {

      if (featureBuffer == null || featureBuffer.length < nRows * nCols) {
        featureBuffer = new Array[Double](nRows * nCols)
      }
      if (labelBuffer == null) {
        labelBuffer = new Array[Double](1)
      }

      // initialize featureBuffer to 0.0
      util.Arrays.fill(featureBuffer, 0, featureBuffer.length, 0.0f)

      val length = math.min(x._1.length, nRows * nCols)
      Array.copy(x._1, 0, featureBuffer, 0, length)
      labelBuffer(0) = x._2

      buffer.set(featureBuffer, labelBuffer,
        Array(nRows, nCols), Array(1))
    })
  }
}

object Test {

  def wrapper(s: String): Seq[String] = {
    s.split(" ").toSeq
  }

  def extractor(s: String): String = {
    s.split("~")(1)
  }

  def main(args: Array[String]): Unit = {
    val coder: (String => String) = (arg: String) => {extractor(arg)}
    val coder2: (String => Seq[String]) = (arg: String) => {wrapper(arg)}
    val sqlfunc = udf(coder)
    val sqlfunc2 = udf(coder2)

    val nodeNum = 1
    val coreNum = 2
    val sc = new SparkContext(Engine.init(nodeNum, coreNum, true).
      get.setMaster("local[2]").setAppName("W2V").set("spark.task.maxFailures", "1"))
    val sqlContext = new SQLContext(sc)

    // DATA PREP
    val path = "PATH_TSV.tsv"
    val tensorBuffer = new ArrayBuffer[LabeledPoint]()
    var i = 0
    while (i < 1000) {
      val input = Tensor[Float](1000).apply1(e => Random.nextFloat())
      val inputArr = input.storage().array()
      tensorBuffer.append(new LabeledPoint(Random.nextInt(10) + 1,
        new DenseVector(inputArr.map(_.toDouble))))
      i += 1
    }
    val rowRDD = sc.parallelize(tensorBuffer)
    val labeled = sqlContext.createDataFrame(rowRDD)

    labeled.show()

    // labeled contains 2 columns "labels" and "vectors"
    // labels are Double.
    // vectors are 1000 in size array of Float
    // row example: ([0.2 0.3 -0.2 .... 0.8], 3.0)
    val vectorizedRdd = labeled.select("label", "features").rdd.map(r =>
      (r(1).asInstanceOf[DenseVector].toArray,
        r(0).asInstanceOf[Double] + 1.0))
    val trainingSplit = 0.6
    val Array(trainingRDD, valRDD) =
      vectorizedRdd.randomSplit(Array(trainingSplit, 1 - trainingSplit))

    val nrows = 10
    val ncols = 100
    val batchSize = 4
    val trainSet = DataSet.rdd(trainingRDD) -> ToSample(nrows, ncols) -> SampleToBatch(batchSize)
    val valSet = DataSet.rdd(valRDD) -> ToSample(nrows, ncols) -> SampleToBatch(batchSize)

    val hiddenSize = 40
    val bpttTruncate = 4
    val inputSize = 100
    val classNum = 34

    val model_N = Sequential[Double]()
      .add(Recurrent[Double](hiddenSize, bpttTruncate)
        .add(RnnCell[Double](inputSize, hiddenSize))
        .add(Tanh[Double]()))
      .add(Select(2, 10))
      // .add(Reshape(Array(400)))
      .add(Linear(40, classNum))
      .add(LogSoftMax())

    val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)

    val optimizer = Optimizer(
      model = model_N,
      dataset = trainSet,
      criterion = new ClassNLLCriterion[Double]()
    ).asInstanceOf[DistriOptimizer[Double]].disableCheckSingleton()

    val numEpochs = 5
    optimizer.
      setState(state).
      setValidation(Trigger.everyEpoch, valSet, Array(new Top1Accuracy[Double])).
      setOptimMethod(new Adagrad[Double]()).
      setEndWhen(Trigger.maxEpoch(numEpochs)).
      optimize()
  }
}
