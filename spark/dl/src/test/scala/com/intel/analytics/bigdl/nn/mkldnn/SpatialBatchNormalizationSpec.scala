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

import com.intel.analytics.bigdl.mkl.{MKL, MklDnn}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.Sequential
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.scalatest.{FlatSpec, Matchers}

class SpatialBatchNormalizationSpec extends FlatSpec with Matchers {
  "bn updateOutput" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 16, 1, 1)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand()
    val initBias = Tensor(channel).rand()

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val input = Tensor(batchSize, channel, height, width).rand()

    bn.weight should be (initWeight)
    bn.bias should be (initBias)

    val output = bn.forward(input)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)
    val nnOutput = nnBn.forward(input)

    output shouldEqual nnOutput
  }

  "bn updateOutput multi times" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val epsilon = 1e-5

    val initWeight = Tensor(channel).rand()
    val initBias = Tensor(channel).rand()

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val input = Tensor(batchSize, channel, height, width).rand()

    bn.weight should be (initWeight)
    bn.bias should be (initBias)

    Utils.manyTimes(bn.forward(input))(10)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    Utils.manyTimes(nnBn.forward(input))(10)

    bn.output should be (nnBn.output)
  }

  "bn backward" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val epsilon = 0.0f

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor().resizeAs(input).rand(-1, 1)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    bn.forward(input)
    nnBn.forward(input)

    val gradInput = bn.backward(input, gradOutput)
    val nnGradInput = nnBn.backward(input, gradOutput)

    bn.gradWeight shouldEqual nnBn.gradWeight
    bn.gradBias shouldEqual nnBn.gradBias
    gradInput should be (nnGradInput)
  }

  "bn backward multi times" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val epsilon = 0.0f

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor().resizeAs(input).rand(-1, 1)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    bn.forward(input)
    nnBn.forward(input)

    Utils.manyTimes(bn.backward(input, gradOutput))(10)
    Utils.manyTimes(nnBn.backward(input, gradOutput))(10)

    bn.gradInput shouldEqual nnBn.gradInput
    bn.gradWeight shouldEqual nnBn.gradWeight
    bn.gradBias shouldEqual nnBn.gradBias
  }

  "bn perf" should "work correctly" in {
    val (batchSize, channel, height, width) = (4, 64, 112, 112)
    val epsilon = 0.0f

    val initWeight = Tensor(channel).rand(-1, 1)
    val initBias = Tensor(channel).rand(-1, 1)

    val bn = SpatialBatchNormalization(channel, epsilon, initWeight = initWeight,
      initBias = initBias)
    val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
    val gradOutput = Tensor().resizeAs(input).rand(-1, 1)

    val nnBn = nn.SpatialBatchNormalization(channel, epsilon,
      initWeight = initWeight, initBias = initBias)

    val times = Utils.manyTimes {
      bn.forward(input)
//      bn.backward(input, gradOutput)
    } _

    val nnTimes = Utils.manyTimes {
      nnBn.forward(input)
//      nnBn.backward(input, gradOutput)
    } _

    times(10)
    nnTimes(10)

    val costs = times(50)._1
    val nnCosts = nnTimes(50)._1

    println(costs)
    println(nnCosts)
  }

  "Convolution + SpatialBarchNormalization" should "work correctly" in {
    MKL.setNumThreads(1)
    import MklDnn.{MemoryFormat => format}
    val dnn = Sequential()
      .add(ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3).setName("conv1/7x7_s2"))
      .add(SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
      .add(MemoryReOrder(inputFormat = format.any, outputFormat = format.nchw))

    val blas = Sequential()
      .add(nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3).setName("conv1/7x7_s2"))
      .add(nn.SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))

    for (i <- dnn.parameters()._1.indices) {
      blas.parameters()._1(i).rand(-1, 1)
      dnn.parameters()._1(i).copy(blas.parameters()._1(i))
    }

    val input = Tensor(4, 3, 224, 224).rand()

    blas.forward(input)
    dnn.forward(input)

    DnnUtils.nearequals(dnn.output.toTensor, blas.output.toTensor)

    val gradOutput = Tensor().resizeAs(blas.output.toTensor).rand()

    blas.backward(input, gradOutput)
    dnn.backward(input, gradOutput)

    DnnUtils.nearequals(dnn.gradInput.toTensor, blas.gradInput.toTensor)
  }
}
