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

import com.intel.analytics.bigdl.dataset.{SampleToBatch, _}
import com.intel.analytics.bigdl.example.MlUtils.RNNParams
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.{DataSet => _}
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
// import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object Test {
  def extractor(s: String): String = {
    s.split("~")(1)
  }

  def main(args: Array[String]): Unit = {
    MlUtils.predictParser.parse(args, RNNParams()).map(param => {

      val coder: (String => String) = (arg: String) => {
        extractor(arg)
      }
      val sqlfunc = udf(coder)

      val scc = Engine.init(param.nodeNumber, param.coreNumber, true).map(conf => {
        conf.setAppName("Predict with trained model")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })
      val sc = scc.get
      val sqlContext = new SQLContext(sc)

      // DATA PREP
      val tensorBuffer = new ArrayBuffer[LabeledPoint]()
      var i = 0
      while (i < 4 * 28 * 400) {
        val input = Tensor[Float](1000).apply1(e => Random.nextFloat())
        val inputArr = input.storage().array()
        tensorBuffer.append(new LabeledPoint(Random.nextInt(10) + 1,
          new DenseVector(inputArr.map(_.toDouble))))
        i += 1
      }
      val rowRDD = sc.parallelize(tensorBuffer)
      val labeled = sqlContext.createDataFrame(rowRDD)

      // labeled contains 2 columns "labels" and "vectors"
      // labels are Double.
      // vectors are 1000 in size array of Float
      // row example: ([0.2 0.3 -0.2 .... 0.8], 3.0)
      val valRDD = labeled.select("label", "features").rdd.map(r =>
        (r(1).asInstanceOf[DenseVector].toArray,
          r(0).asInstanceOf[Double] + 1.0))

      val nrows = 10
      val ncols = 100
      val batchSize = param.batchSize
      val valSet = DataSet.rdd(valRDD) -> ToSample(nrows, ncols) -> SampleToBatch(batchSize)

      val path_saveModel = "model_test"
      val loaded_model = Module.load[Double](path_saveModel)

      val tmpRDD = valSet.toDistributed().data(train = false)
      val tmp1 = tmpRDD.mapPartitions(dataIter => {
        dataIter.map(batch => {
          val input = batch.data
          loaded_model.forward(input).toTensor[Double]
        })
      }).collect()

      // tmp1.map(println(_))

      val validator = Validator(loaded_model, valSet)
      val result = validator.test(Array(new Top1Accuracy[Double]))

      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })

    })
  }
}
