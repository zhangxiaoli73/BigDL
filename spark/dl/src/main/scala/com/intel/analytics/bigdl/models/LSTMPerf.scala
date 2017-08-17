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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.numeric.NumericFloat

object GRU {
  def apply(classNum: Int, inputSize: Int, hiddenSize: Int): Module[Float] = {
    val model = nn.Sequential()
    model.add(nn.Recurrent()
      .add(nn.GRU(inputSize, hiddenSize)))
      .add(nn.Select(2, -1))
    model
  }
}

object ConvLSTMPeephole {
  def apply(classNum: Int, inputSize: Int, hiddenSize: Int, kernelC: Int, kernelI: Int, stride: Int)
  : Module[Float] = {
    val model = nn.Sequential()
    model.add(nn.Recurrent()
      .add(nn.ConvLSTMPeephole(inputSize, hiddenSize, kernelC, kernelI, stride)))

    model
  }
}

object SimpleRNN {
  def apply(classNum: Int, inputSize: Int, hiddenSize: Int): Module[Float] = {
    val model = nn.Sequential()
    model.add(nn.Recurrent()
      .add(nn.RnnCell(inputSize, hiddenSize, nn.Tanh[Float]())))
      .add(nn.Select(2, -1))

    model
  }
}

object LSTM {
  def apply(classNum: Int, inputSize: Int, hiddenSize: Int): Module[Float] = {
    val model = nn.Sequential()
    val embedding = 1
    model.add(nn.Recurrent().add(nn.LSTM(inputSize, hiddenSize)))
      .add(nn.Select(2, -1))
    model
  }
}


object LSTMPeephole {
  def apply(classNum: Int, inputSize: Int, hiddenSize: Int): Module[Float] = {
    val model = nn.Sequential()
    val embedding = 1
    model.add(nn.Recurrent().add(nn.LSTMPeephole(inputSize, hiddenSize)))
      .add(nn.Select(2, -1))

    model
  }
}
