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

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.mkldnn.Utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.scalatest.{FlatSpec, Matchers}

class PerfSpec extends FlatSpec with Matchers {
  val core = Runtime.getRuntime.availableProcessors() / 2
  "AlexNet 4-models 1-omp-thread with local optimizer" should "work correctly" in {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", s"$core")
    Engine.init

    val batchSize = 16
    val model = AlexNet(1000)
    println(model)
    val criterion = ClassNLLCriterion()
    val miniBatch = MiniBatch[Float](Tensor(batchSize, 3, 227, 227), Tensor(batchSize).fill(1))

    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          private val index = new AtomicInteger()
          override def hasNext: Boolean = {
            if (train) {
              true
            } else {
              index.get() < 100000
            }
          }

          override def next(): MiniBatch[Float] = {
            index.getAndIncrement()
            miniBatch
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    model.training()
    model.resetTimes()
    val optimizer = Optimizer(model, dummyDataSet, criterion)
    val start = System.nanoTime()
    val optimizedModel = optimizer.setEndWhen(Trigger.maxIteration(50)).optimize()
    val end = System.nanoTime()
    println((end - start) / 1e9)
    println("AlexNet 4-models 1-omp-thread with local optimizer " +
      s"throughput = ${16 * 50 / ((end - start).toDouble / 1e9)}")
  }

  "AlexNet 1-models 4-omp-thread with local optimizer" should "work correctly" in {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", "1")
    System.setProperty("bigdl.mklNumThreads", s"$core")
    Engine.init

    val batchSize = 16
    val model = AlexNet(1000)
    println(model)
    val criterion = ClassNLLCriterion()
    val miniBatch = MiniBatch[Float](Tensor(batchSize, 3, 227, 227), Tensor(batchSize).fill(1))

    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          private val index = new AtomicInteger()
          override def hasNext: Boolean = {
            if (train) {
              true
            } else {
              index.get() < 100000
            }
          }

          override def next(): MiniBatch[Float] = {
            index.getAndIncrement()
            miniBatch
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    model.training()
    model.resetTimes()
    val optimizer = Optimizer(model, dummyDataSet, criterion)
    val start = System.nanoTime()
    val optimizedModel = optimizer.setEndWhen(Trigger.maxIteration(50)).optimize()
    val end = System.nanoTime()
    println((end - start) / 1e9)
    println("AlexNet 1-models 4-omp-thread with local optimizer " +
      s"throughput = ${16 * 50 / ((end - start).toDouble / 1e9)}")
  }

  "AlexNet 1-model 4-omp-threads current thread" should "work correctly" in {
    System.setProperty("bigdl.mklNumThreads", s"$core")
    val batchSize = 16
    val model = AlexNet(1000)
    model.createDnnEngine(0)
    model.createStream()
    model.training()

    val input = Tensor(batchSize, 3, 227, 227).rand()
    val gradOutput = Tensor(batchSize, 1000).rand()
    model.resetTimes()

    val warm = 10
    val iters = 50

    val time = manyTimes {
      model.forward(input)
      model.backward(input, gradOutput)
    } _

    val (_, _) = time(warm)
    model.resetTimes()
    val taken = time(iters)._1

    println(taken / iters)
    println("AlexNet 1-model 4-omp-threads current thread " +
      s"throughput = ${16 * iters / taken * 1000}")
  }

  "AlexNet 1-model 4-omp-threads with invokeAndWait" should "work correctly" in {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", "1")
    System.setProperty("bigdl.mklNumThreads", s"$core")
    Engine.init

    val batchSize = 16
    val model = AlexNet(1000)
    model.createDnnEngine(0)
    model.createStream()
    model.training()

    val input = Tensor(batchSize, 3, 227, 227).rand()
    val gradOutput = Tensor(batchSize, 1000).rand()
    model.resetTimes()

    val warm = 10
    val iters = 50

    val time = manyTimes {
      Engine.default.invokeAndWait(
        (0 until 1).map(i =>
          () => {
            model.forward(input)
            model.backward(input, gradOutput)
            1
          }))
    } _

    val (_, _) = time(warm)
    model.resetTimes()
    val taken = time(iters)._1

    println(taken / iters)
    println("AlexNet 1-model 4-omp-threads with invokeAndWait " +
      s"throughput = ${16 * iters / taken * 1000}")
  }

  "AlexNet 4-models 1-omp-thread with invokeAndWait" should "work correctly" in {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", s"$core")
    System.setProperty("bigdl.mklNumThreads", "1")
    Engine.init

    val batchSize = 4 // should not 16
    val model = AlexNet(1000)
    model.createDnnEngine(0)
    model.createStream()
    model.training()

    val input = Tensor(batchSize, 3, 227, 227).rand()
    val gradOutput = Tensor(batchSize, 1000).rand()
    model.resetTimes()

    val models = Array.fill(4)(model.cloneModule())
    for (i <- 0 until 4) {
      models(i).createDnnEngine(0)
      models(i).createStream()
      models(i).training()
    }

    val warm = 10
    val iters = 50

    val time = manyTimes {
      Engine.default.invokeAndWait(
        (0 until 4).map(i =>
          () => {
            models(i).forward(input)
            models(i).backward(input, gradOutput)
            1
          }))
    } _

    val (_, _) = time(warm)
    model.resetTimes()
    val taken = time(iters)._1

    println(taken / iters)
    println("AlexNet 4-models 1-omp-thread with invokeAndWait " +
      s"throughput = ${16 * iters / taken * 1000}")
  }
}
