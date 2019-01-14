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

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{CropCenter, CropRandom, CropperMethod, LabeledBGRImage}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.visualization.ValidationSummary
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.Iterator

class UserTest(sparkContext: SparkContext) {

  val random = scala.util.Random

  def createModel(): Sequential[Float] = {
    import com.intel.analytics.bigdl.nn._
    val model = Sequential[Float]()

    val parallel = ParallelTable()
    val m1 = Sequential[Float]().add(Identity[Float]()).add(Echo())
    val m2 = Sequential[Float]().add(Identity[Float]()).add(Echo())
    parallel.add(m1).add(m2)
    model.add(parallel).add(JoinTable(2, 2)).add(InferReshape(Array(-1, 10, 13)))

    val recurrent = Recurrent[Float]()
    recurrent.add(RnnCell(13, 7, Tanh()))

    model.add(recurrent)
      .add(TimeDistributed(Linear(7, 2)))
      .add(TimeDistributed(LogSoftMax()))

    model
  }

  def createSamples(numSamples: Int): RDD[Sample[Float]] = {
    val samples = Array.tabulate(numSamples) { _ =>
      val x1 = Tensor(10, 8).rand()
      val x2 = Tensor(10, 5).rand()
      val features = Array(x1, x2)
      val label = Tensor(10).fill(1f)
      Sample(features, label)
    }
    sparkContext.parallelize(samples)
  }

  def createTensorSamples(numSamples: Int): RDD[Sample[Float]] = {
    val samples = Array.tabulate(numSamples) { _ =>
      val x1 = Tensor(10, 8).rand()
      val x2 = Tensor(10, 5).rand()

      val features = Array(Tensor.sparse(x1), Tensor.sparse(x2))
      val label = Tensor.sparse(Tensor(10).fill(1f))
      TensorSample(features, label)
    }
    sparkContext.parallelize(samples)
  }

}

object SparseToDense {
  def apply(): SparseToDense = new SparseToDense()
}

class SparseToDense() extends Transformer[Sample[Float], Sample[Float]] {
  private var featrues: Array[Tensor[Float]] = null
  private var labels: Array[Tensor[Float]] = null

  override def apply(prev: Iterator[Sample[Float]]): Iterator[Sample[Float]] = {
    prev.map(data => {
      val img = data.asInstanceOf[TensorSample[Float]]
      if (featrues == null) featrues = new Array[Tensor[Float]](img.features.length)
      if (labels == null) labels = new Array[Tensor[Float]](img.labels.length)

      var i = 0
      while (i < img.features.length) {
        if (featrues(i) == null) featrues(i) = Tensor[Float](img.features(i).size())
        Tensor.dense(img.features(i), featrues(i))
        i += 1
      }

      i = 0
      while (i < img.labels.length) {
        if (labels(i) == null) labels(i) = Tensor[Float](img.labels(i).size())
        Tensor.dense(img.labels(i), labels(i))
        i += 1
      }
      Sample(featrues, labels)
    })
  }
}

object UserTest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf()
      .setMaster("local[1]")
      .setAppName("Batch")
      .set("spark.task.maxFailures", "1")
      .set("spark.driver.host", "localhost")
    val sparkContext = new SparkContext(conf)
    Engine.init
    val app = new UserTest(sparkContext)

    val rdd = app.createTensorSamples(100)
    val batchSize = 8



    val tmp = rdd.count()

    val model = app.createModel()
    val paddingX = PaddingParam[Float](Some(Array(Tensor(T(0f)), Tensor(T(0f)))))
    val paddingY = PaddingParam[Float](Some(Array(Tensor(T(0f)))))

    val dataset = (DataSet.rdd(rdd) ->
      SparseToDense() ->
      SampleToMiniBatch(batchSize, Some(paddingX), Some(paddingY)))
      .asInstanceOf[DistributedDataSet[MiniBatch[Float]]]

    val tmp111 = dataset.data(train = false).take(1)

    val optimizer = Optimizer(
      model,
      dataset,
      TimeDistributedCriterion(ClassNLLCriterion[Float](paddingValue = 0), true)
    )

    val summary = ValidationSummary(appName = "Batch", logDir = "/tmp/")

    optimizer.setOptimMethod(new Adagrad[Float](learningRate = 1E-2, learningRateDecay = 1E-3))
      .setEndWhen(Trigger.maxEpoch(10))
      .setValidationSummary(summary)
      .optimize()

  }
}


