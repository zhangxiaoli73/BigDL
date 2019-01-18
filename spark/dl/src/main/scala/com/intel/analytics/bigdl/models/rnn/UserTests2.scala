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

package com.intel.analytics.bigdl.models.rnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{PaddingParam, Sample}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.visualization.ValidationSummary
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class UserTestNew(sparkContext: SparkContext) {

  val random = scala.util.Random

  def createModel(): Module[Float] = {
    val model = Sequential[Float]()

    val parallel = ParallelTable().setName("parallelTable")
    val ws = Sequential[Float]()
      .add(DenseToSparse(propagateBack = false))
      .add(LookupTableSparse[Float](1000 + 1, 100, "mean"))

    val wordModule = Sequential[Float]()
      .add(SplitTable(1, 3))
      .add(MapTable(ws))
      .add(Pack(1))
      .add(Echo())

    val cs = Sequential[Float]()
      .add(DenseToSparse(propagateBack = false))
      .add(LookupTableSparse[Float](150 + 1, 50, "mean"))

    val charModule = Sequential[Float]()
      .add(SplitTable(1, 3))
      .add(MapTable(cs))
      .add(Pack(1))
      .add(Echo())


    // model.add(parallel).add(JoinTable(2, 2)).add(Echo()).inputs()

    val node1 = parallel.add(wordModule).add(charModule).setName("parallel").inputs()
    val node2 = JoinTable(2, 2).inputs(node1)
    val node3 = Echo().inputs(node2)
    val rnn = Recurrent[Float]().add(RnnCell(100 + 50, 16, Tanh())).inputs(node3)
    val node4 = TimeDistributed(Linear(16, 2)).inputs(rnn)
    val node5 = TimeDistributed(LogSoftMax()).inputs(node4)

//    model.add(rnn)
//      .add(TimeDistributed(Linear(16, 2)))
//      .add(TimeDistributed(LogSoftMax()))

    val m = Graph(node1, node5)
    m.stopGradient(Array("parallel"))
    m
  }

  def createSamples(numSamples: Int): RDD[Sample[Float]] = {
    val samples = Array.tabulate(numSamples){ _ =>
      val n = random.nextInt(40)
      // generate random x1 of indices in the range [0, 1000] (word vocab)
      val x1 = Tensor(n, 1).fill(0f)
      for (i <- 0 until n)
        x1.setValue(i + 1, 1, 1 + random.nextInt(1000))
      // generate a random x2 of indices in the range [0, 150] (char vocab)
      // suppose that each row of x2 contains only a small random number of indices corresponding
      // to the number of characters of a word (we have 20 words)
      val x2 = Tensor(n, 150).fill(1f)
      for (i <- 0 until n)
        for (j <- 0 until random.nextInt(15))
          x2.setValue(i + 1, j + 1, 1 + random.nextInt(150))

      val features = Array(x1, x2)
      // generate random binary label for testing
      val label = Tensor(n).zero()
      for (i <- 0 until n)
        label.setValue(i + 1, random.nextInt(2))
      Sample(features, label)
    }
    sparkContext.parallelize(samples)
  }

}


object UserTestNew {


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf()
      .setMaster("local[1]")
      .setAppName("Batch")
      .set("spark.task.maxFailures", "1")
      .set("spark.driver.host", "localhost")
    val sparkContext = new SparkContext(conf)
    Engine.init
    val app = new UserTestNew(sparkContext)


    val rdd = app.createSamples(100)
    val model = app.createModel()
    val batchSize = 8
    val paddingX = PaddingParam[Float](Some(Array(Tensor(T(1f)), Tensor(T(1f)))))

    val paddingY = PaddingParam[Float](Some(Array(Tensor(T(0f)))))

    val optimizer = Optimizer(
      model = model,
      sampleRDD = rdd,
      criterion = TimeDistributedCriterion(ClassNLLCriterion[Float](paddingValue = 0), true),
      batchSize = batchSize,
      featurePaddingParam = paddingX,
      labelPaddingParam = paddingY
    )

    val summary = ValidationSummary(appName = "Batch", logDir = "/tmp/")

    optimizer.setOptimMethod(new Adagrad[Float](learningRate = 1E-2, learningRateDecay = 1E-3))
      .setEndWhen(Trigger.maxEpoch(10))
      .setValidationSummary(summary)
      .optimize()

  }
}