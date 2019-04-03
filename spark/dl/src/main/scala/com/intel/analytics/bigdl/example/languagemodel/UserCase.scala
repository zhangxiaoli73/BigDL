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

package com.intel.analytics.bigdl.example.languagemodel


import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.dataset.{FixedLength, PaddingParam}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache
import org.apache.spark
import org.apache.spark.SparkContext

object UserCase {
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Train ptbModel on text")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init
    val vocabNum = 5000
    val embeddingDim = 50
    val maxSequenceLength = 1000
    val categoryNum = 16
    val executorCores = 4
    val numExecutors = 1
    val batchSize = executorCores*numExecutors

    val preprocessor = Sequential().add(
      LookupTable(vocabNum, embeddingDim).setInitMethod(Xavier).freeze())
    val model: Sequential[Float] = preprocessor  // input: Tensor of size(batchSize, SeqLength, EmbeddingDim)
    //model.add(Masking(0))// mask with value 0 for time-status
    model.add(Recurrent().add(GRU(embeddingDim, 16)))
    model.add(Select(2, maxSequenceLength)) // take the last time-step status which is at dim 2, these two params are count from 1
    model.add(Linear(16, categoryNum))
    model.add(SoftMax())

    println(model.toString())
    print(s"\n\n================== Test Forward======================\n\n")
    val testSeq = Seq.fill(maxSequenceLength)(3f).toArray
    testSeq(0) = 2f

    /*
     You should run next one code twice
    */



    val cloneModel = model.cloneModule()
    val testTensor = Tensor(testSeq, Array(1,maxSequenceLength))
    // println(model.forward(testTensor).toTensor)

    print(s"\n\n===================== Test Evaluate==================\n\n")
    val testLabel = new Array[Float](1)
    testLabel(0) = categoryNum
    val featureTensor = Tensor(testSeq, Array(maxSequenceLength))
    val labelTensor = Tensor(testLabel, Array(1))
    val testSample = Sample(featureTensor, labelTensor)
    val testSampleMulti = Sample(Tensor().resizeAs(featureTensor).copy(featureTensor),
      Tensor().resizeAs(labelTensor).copy(labelTensor))
    val testSet = sc.parallelize(Seq(testSample))
    println(model.evaluate(testSet, Array(new Top1Accuracy[Float]))(0))

    print(s"\n\n======================== Test 4096 Sample Evaluate====================\n\n")
    val testSetMulti = sc.parallelize(Seq.fill(4096)(testSample)).coalesce(numExecutors)
    println(model.evaluate(testSetMulti, Array(new Top1Accuracy[Float]))(0))

//    val out = model.forward(testTensor)
//    val grad = model.backward(testTensor, Tensor(1, 16).rand().asInstanceOf[Activity])

    print(s"\n\n============== Test training==============\n\n")
    val optimizer = Optimizer(model = model,
      sampleRDD = testSetMulti, criterion = ClassNLLCriterion[Float](), batchSize = batchSize)

    optimizer.setOptimMethod(new ParallelAdam[Float]()).setEndWhen(Trigger.severalIteration(10))
    optimizer.optimize()

    println("====================== Test done =====================")
  }
}
