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

import com.intel.analytics.bigdl.dataset.{SampleToBatch, Transformer, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adagrad, _}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{StringIndexer, Word2Vec}
import org.apache.spark.rdd._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
// import org.apache.spark.ml.linalg.DenseVector
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.{DataSet => _, _}
import org.apache.spark.ml.linalg.DenseVector

// PropertyConfigurator.configure("/PATH/log4j.properties")

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
      if (labelBuffer == null || labelBuffer.length < nRows) {
        labelBuffer = new Array[Double](nRows)
      }

      var i = 0
      while (i < nRows) {
        Array.copy(x._1, 0, featureBuffer, i * nCols, nCols)
        labelBuffer(i) = x._2
        i += 1
      }

      buffer.copy(featureBuffer, labelBuffer,
        Array(nRows, nCols), Array(nRows))
    })
  }
}

object Test {

  def wrapper(s: String): Seq[String] = {
    var t = s.split(" ").toSeq
    return t
  }

  def extractor(s: String): String = {
    val tmp = s.split("~")
    var t = tmp(1)
    return t
  }

  def main(args: Array[String]): Unit = {
    val coder: (String => String) = (arg: String) => {extractor(arg)}
    val coder2: (String => Seq[String]) = (arg: String) => {wrapper(arg)}
    val sqlfunc = udf(coder)
    val sqlfunc2 = udf(coder2)

    val nodeNum = 1
    val coreNum = 1
    val sc = new SparkContext(Engine.init(nodeNum, coreNum, true).
      get.setMaster("local[1]").setAppName("W2V").set("spark.task.maxFailures", "1"))
    val sqlContext = new SQLContext(sc)

    val path = "PATH_TSV.tsv"
    val data = sqlContext.read.format("com.databricks.spark.csv").
      option("header", "true").option("delimiter", "\t").load(path)

    data.printSchema()
    data.show()

    val data_vectorized = data.withColumn("phraseArray", sqlfunc2(col("phrase"))).
      withColumn("level1", sqlfunc(col("cat")))

    data_vectorized.show()

    val indexer = new StringIndexer().setInputCol("level1").setOutputCol("level1_labels")
    val labeled = indexer.fit(data_vectorized).transform(data_vectorized)

    val embeddingDim = 300
    val word2Vec = new Word2Vec().setSeed(32).setInputCol("phraseArray").
      setOutputCol("vectors").setVectorSize(embeddingDim).setMinCount(0).setWindowSize(5)
    val model_w2v = word2Vec.fit(labeled)
    val result = model_w2v.transform(labeled)
    val trainingSplit: Double = 0.8
    val numClass = 34


    val vectorizedRdd : RDD[(Array[Double], Double)] = result.select("level1_labels", "vectors").
      rdd.map(r => (r(1).asInstanceOf[DenseVector].toArray, r(0).asInstanceOf[Double]))
    val Array(trainingRDD, valRDD) =
      vectorizedRdd.randomSplit(Array(trainingSplit, 1 - trainingSplit))

    val batchSize = 9
    val trainSet = DataSet.rdd(trainingRDD) -> ToSample(1, 2) -> SampleToBatch(batchSize)
    val valSet = DataSet.rdd(valRDD) -> ToSample(1, 2) -> SampleToBatch(batchSize)

    val classNum = 32
    val model_N = Sequential()
      .add(Reshape(Array(2)))
      .add(Linear(2, classNum))
      // .add(LogSoftMax())

    val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)

    val criterion = new CrossEntropyCriterion[Double](squeezeFlag = true)
    val optimizer = Optimizer(
      model = model_N,
      dataset = trainSet,
      criterion = criterion.asInstanceOf[Criterion[Double]]
    )

    optimizer.
      setState(state).
      setValidation(Trigger.everyEpoch,
        // valSet, Array(new Top1Accuracy[Double], new Top5Accuracy[Double])).
        valSet, Array(new Loss[Double])).
      setOptimMethod(new Adagrad[Double]()).
      optimize()
  }
}


