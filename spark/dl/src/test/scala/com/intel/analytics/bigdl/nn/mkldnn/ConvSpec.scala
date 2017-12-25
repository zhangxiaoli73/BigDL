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

import com.intel.analytics.bigdl.nn.{ReLU, SpatialConvolution, SpatialFullConvolution}
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.spark.sql.catalyst.expressions.Conv
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class ConvSpec extends FlatSpec with Matchers {

  def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
    val absA = math.abs(a)
    val absB = math.abs(b)
    val diff = math.abs(a - b)

    val result = if (a == b) {
      true
    } else if (a == 0 || b == 0 || diff < java.lang.Float.MIN_NORMAL) {
      diff < (epsilon * java.lang.Float.MIN_NORMAL)
    } else {
      diff / (absA + absB) < epsilon
    }

    result
  }

    def nearequals(t1: Tensor[Float], t2: Tensor[Float]): Boolean = {
      var result = true
      t1.map(t2, (a, b) => {
        if (result) {
          result = nearlyEqual(a, b, DenseTensorMath.floatEpsilon)
        }
        a
      })
      return result
    }
    "Conv dnn forward" should "work correctly" in {
      val nInputPlane = 2
      val nOutputPlane = 4
      val kW = 3
      val kH = 3
      val dW = 4
      val dH = 4
      val padW = 0
      val padH = 0

      Random.setSeed(100)
      val input = Tensor[Float](2, 2, 23, 23).apply1(e => 1.0f)
      val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => 1.0f)
      RNG.setSeed(100)
      val conv = Conv(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      RNG.setSeed(100)
      val layer = SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

      // warm up
      val output = conv.forward(input)
      val grad1 = conv.updateGradInput(input, gradOutput)
      conv.accGradParameters(input, gradOutput)
      val weight1 = conv.weight
      val gradweight1 = conv.gradWeight
      val bias1 = conv.bias
      val gradbias1 = conv.gradBias
      val output2 = layer.forward(input)
      val grad2 = layer.updateGradInput(input, gradOutput)
      layer.accGradParameters(input, gradOutput)
      val weight2 = layer.weight
      val gradweight2 = layer.gradWeight
      val bias2 = conv.bias
      val gradbias2 = conv.gradBias

      // weight1 should be(weight2)
      nearequals(weight1, weight2) should be(true)
      nearequals(gradweight1, gradweight2) should be(true)
      nearequals(bias1, bias2) should be(true)
      nearequals(gradbias1, gradbias2) should be(true)
      nearequals(output, output2) should be(true)
      nearequals(grad1, grad2) should be(true)

     println("done")
    // input should be (output)
  }

  "Conv forward" should "work correctly" in {
    val nInputPlane = 3
    val nOutputPlane = 96
    val kW = 11
    val kH = 11
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0

    val input = Tensor[Float](1, 3, 227, 227).apply1(e => Random.nextFloat())
    val layer = new SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
    // warm up
    for (i <- 1 to 10) {
      val output = layer.forward(input)
    }

    val s1 = System.nanoTime()
    for (i <- 1 to 50) {
      val output = layer.forward(input)
    }
    val end1 = System.nanoTime() - s1
    println(s"conv time ${end1/1e9}")
    println("done")
    // input should be (output)
  }

  "Conv dnn 111 forward" should "work correctly" in {
    val nInputPlane = 3
    val nOutputPlane = 96
    val kW = 11
    val kH = 11
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0

    val input = Tensor[Float](1, 3, 227, 227).apply1(e => Random.nextFloat())

    val layer = Conv(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    // warm up
    for (i <- 1 to 10) {
      val output = layer.forward(input)
    }

    val s1 = System.nanoTime()
    for (i <- 1 to 50) {
      val output = layer.forward(input)
    }
    val end1 = System.nanoTime() - s1
    println(s"conv time ${end1/1e9}")
    println("done")
    // input should be (output)
  }
}
