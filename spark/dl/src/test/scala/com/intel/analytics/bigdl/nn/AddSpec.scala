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

package com.intel.analytics.bigdl.nn

import breeze.linalg.{*, dim}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.{TensorCriterion, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import spire.syntax.rng

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class AddSpec extends FlatSpec with Matchers {

  "A Add with scaleB" should "work correctly" in {
    val inputN = 5
    val seed = 100
    RNG.setSeed(seed)
    val layer1 = new Add[Double](inputN)
    val layer2 = layer1.cloneModule().asInstanceOf[Add[Double]]
      .setScaleB(2.0)

    val input = Tensor[Double](1, 5)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 3
    input(Array(1, 4)) = 4
    input(Array(1, 5)) = 5
    val gradOutput = Tensor[Double](5)
    gradOutput(Array(1)) = 2
    gradOutput(Array(2)) = 5
    gradOutput(Array(3)) = 10
    gradOutput(Array(4)) = 17
    gradOutput(Array(5)) = 26

    val output1 = layer1.forward(input)
    val gradInput1 = layer1.backward(input, gradOutput)
    val output2 = layer2.forward(input)
    val gradInput2 = layer2.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)

    layer2.gradBias should be (layer1.gradBias.mul(2))
  }

  "test " should "more better" in {
    val batch = 200

    val input1 = Tensor[Float](batch, 1500).randn()
    val weight1 = Tensor[Float](1500, 1).randn()
    val weight2 = Tensor[Float](1, 10000).randn()

    val output1 = Tensor[Float](batch, 1)
    val output2 = Tensor[Float](batch, 10000)

    val t11 = System.nanoTime()
    for (i <- 1 to 100) {
      output1.mm(input1, weight1)
      output2.mm(output1, weight2)
    }
    val end11 = System.nanoTime() - t11
    println("time: " + end11/1e9 + " s")

    // ************

    val input = Tensor[Float](batch, 1500).randn()
    val weight = Tensor[Float](1500, 10000).randn()

    val output = Tensor[Float](batch, 10000).rand()

    val t1 = System.nanoTime()
    for (i <- 1 to 100) {
      output.mm(input, weight)
    }
    val end1 = System.nanoTime() - t1
    println("time: " + end1/1e9 + " s")

    // ***************

  }


   "111" should "222" in {
     val inputSize = 650 // param.inputSize
     val hiddenSize = 6000 // param.hiddenSize
     val batchSize = 2

     val input = Tensor[Float](Array(batchSize, inputSize)).fill(2.0f)
     val labels = Tensor[Float](Array(batchSize, hiddenSize)).fill(1)

     RNG.setSeed(100)
     val model1 = Linear[Float](inputSize, hiddenSize)
     RNG.setSeed(100)
     val model2 = Sequential[Float]().
       add(Linear[Float](inputSize, 100)).add(Linear[Float](100, hiddenSize))

     // warm up
     for (i <- 1 to 100) {
       val out2 = model2.forward(input)
       val grad = model2.backward(input, labels)
     }
     for (i <- 1 to 100) {
       val out1 = model1.forward(input)
       val grad = model1.backward(input, labels)
     }
     // ****************
     val t1 = System.nanoTime()
     for (i <- 1 to 100) {
       val out1 = model1.forward(input)
       val grad = model1.backward(input, labels)
     }
     val end1 = System.nanoTime() - t1

     val t2 = System.nanoTime()
     for (i <- 1 to 100) {
       val out2 = model2.forward(input)
       val grad = model2.backward(input, labels)
     }
     val end2 = System.nanoTime() - t2

     println(s"end1 ${end1/1e9} end2 ${end2/1e9}")

     val grad1 = model1.getParameters()
     val grad2 = model2.getParameters()

     println("done")
  }

  "333" should "444" in {
    RNG.setSeed(100)
    val tmp1 = Tensor[Float](10, 10).apply1(e => RNG.random())
    val tmp2 = Tensor[Float](10, 10).apply1(e => RNG.random())
    val tmp22 = Tensor[Float](10, 10).apply1(e => RNG.random())
    val tmp3 = Tensor[Float](10, 3, 10)
    val tmp4 = Tensor[Float](10, 3, 10)

    val times = 3
    val timeDim = 2
    val length3 = 10
    val batchDim = 10

    val src = Array(tmp1, tmp2, tmp22)
    def copy(src: Array[Tensor[Float]], dst: Tensor[Float], offset: Int): Unit = {
      var t = 1
      while ((t + offset) <= times) {
        dst.select(timeDim, t + offset).copy(src(t - 1))
        t += 1
      }
    }

    def copy2(src: Array[Tensor[Float]], dst: Tensor[Float], offset: Int): Unit = {
      var t = 1
      var tt = 0
      val dstArr = dst.storage().array()
      while (t <= times) {
//        dst.select(timeDim, t + offset).copy(src(t - 1))
        val t1 = src(t - 1).storage().array()
        var l = 0
        while (l < batchDim) {
          System.arraycopy(t1, l * length3, dstArr, times * length3 * l + length3 * (t-1), length3)
          l += 1
        }
        t += 1
      }
    }

    var start1 = System.nanoTime()
    for (i <- 1 to 100) {
      copy(src, tmp3, 0)
    }
    var end1 = System.nanoTime() - start1

    println(tmp1)
    println(tmp2)
    println("src")
    println(tmp3)
    println("src 222")
    var start2 = System.nanoTime()
    for (i <- 1 to 100) {
      copy2(src, tmp4, 0)
    }
    var end2 = System.nanoTime() - start2
    println(tmp4)

    tmp4 should be (tmp3)
    println("end1 " + end1/1e9 + " end2 " + end2/1e9)
  }

  "333 111111" should "444" in {
    RNG.setSeed(100)
    val tmp1 = Tensor[Float](30, 10, 128, 10).apply1(e => Random.nextFloat())
    var tmp3 = Tensor[Float](10, 30, 128, 10)
    var tmp4 = Tensor[Float](10, 30, 128, 10)

    val times = 30
    val length3 = 128*10
    val batchDim = 10

    def copy2(src: Tensor[Float], dst: Tensor[Float], offset: Int): Unit = {
      var t = 1
      var tt = 0
      var l = 0
      val dstArr = dst.storage().array()
      val t1 = src.storage().array()
      val length4 = batchDim * length3
      while (t <= times) {
        var l = 0
        val length1 = batchDim * length3 * (t-1)
        val length2 = (t-1) * length3
        while (l < length4) {
          System.arraycopy(t1, length1 + l, dstArr, l * times + length2, length3)
          l += length3
        }
        t += 1
      }
    }

    def transposeMemory(src: Tensor[Float], dst: Tensor[Float]): Unit = {
      var t = 1
      val dstArr = dst.storage().array()
      val srcArr = src.storage().array()

      val firstSize = src.size(1)
      val secondSize = src.size(2)
      val otherSize = src.nElement() / (firstSize * secondSize)

      val length3 = secondSize * otherSize
      while (t <= firstSize) {
        var l = 0
        val length1 = secondSize * otherSize * (t-1)
        val length2 = (t-1) * otherSize
        while (l < length3) {
          System.arraycopy(srcArr, length1 + l, dstArr, l * firstSize + length2, otherSize)
          l += otherSize
        }
        t += 1
      }
    }

    var start2 = System.nanoTime()
    for (i <- 1 to 100) {
      transposeMemory(tmp1, tmp4)
    }
    var end2 = System.nanoTime() - start2

    var start1 = System.nanoTime()
    for (i <- 1 to 100) {
      val buffer = tmp1.transpose(1, 2)
      tmp3.resizeAs(buffer).copy(buffer)
    }
    var end1 = System.nanoTime() - start1

    println(tmp1)
    println("src")
    println(tmp3)
    println("src 222")

    println(tmp4)

    tmp4 should be (tmp3)
    println("end1 " + end1/1e9 + " end2 " + end2/1e9)
  }

  "333 2222222222" should "444" in {
    RNG.setSeed(100)
    val dim = 2
    val index = 2
    val tmp1 = Tensor[Float](3, 2, 4).apply1(e => Random.nextFloat())
    val tmp3 = Tensor[Float]()
    var tmp2 = Tensor[Float]()

    def copyMemory(src: Tensor[Float], dst: Tensor[Float], index: Int): Unit = {
      val srcSize = src.size()
      val batchSize = srcSize(0)
      val timeSize = srcSize(1)
      val otherSize = src.nElement() / (batchSize * timeSize)
      val srcArr = src.storage().array()
      val srcOffset = src.storageOffset() - 1

      srcSize(0) = timeSize
      srcSize(1) = batchSize
      dst.resize(srcSize)
      val dstArr = dst.storage().array()
      val dstOffset = dst.storageOffset() - 1

      var t = 1
      val l = (index-1) * otherSize
      while (t <= batchSize) {
        val length1 = timeSize * otherSize * (t-1) + srcOffset
        val length2 = (t-1) * otherSize + dstOffset
        System.arraycopy(srcArr, length1 + l, dstArr, l * batchSize + length2, otherSize)
        t += 1
      }
    }
    println(tmp1)
    println("src")
    var tmp = tmp1.select(dim, index)
    println(tmp)

    var start2 = System.nanoTime()
    for (i <- 1 to 1000) {
      copyMemory(tmp1, tmp3, index)
    }
    var end2 = System.nanoTime() - start2

    var start1 = System.nanoTime()
    for (i <- 1 to 1000) {
      tmp2 = tmp.resizeAs(tmp).copy(tmp)
    }
    var end1 = System.nanoTime() - start1


    println(tmp3)
    val tmp4 = tmp3.select(1, index)
    println(tmp4)

    tmp2 should be (tmp4)
    println("end1 " + end1/1e9 + " end2 " + end2/1e9)
    println("done")
  }

  "333 33333" should "444" in {
    RNG.setSeed(100)
    val dim = 2
    val index = 2
    val tmp1 = Tensor[Float](3, 2, 4).apply1(e => Random.nextFloat())
    val tmp3 = Tensor[Float](3, 2, 4)
    var tmp2 = Tensor[Float](3, 4).apply1(e => Random.nextFloat())
    val tmp4 = Tensor[Float](3, 2, 4)


    tmp3.select(dim, index).copy(tmp2)

    def copyMemory(src: Tensor[Float], dst: Tensor[Float], dstDim: Int, dstIndex: Int): Unit = {
      val dstArr = dst.storage().array()
      val dstOffset = dst.storageOffset() - 1
      val batchSize = dst.size(1)
      val times = dst.size(2)
      val otherSize = dst.nElement() / (batchSize * times)

      val length2 = batchSize * otherSize
      val srcArr = src.storage().array()
      val srcOffset = src.storageOffset() - 1
      val length1 = (dstIndex - 1) * otherSize + dstOffset
      var l = 0
      while (l < length2) {
        System.arraycopy(srcArr, l + srcOffset, dstArr, times * l + length1, otherSize)
        l += otherSize
      }
    }

    copyMemory(tmp2, tmp4, dim, index)

    println(tmp2)
    println("src")
    println(tmp3)
    println("11111111111111111")
    println(tmp4)

    // tmp3 should be (tmp4)
    println("done")
  }
}
