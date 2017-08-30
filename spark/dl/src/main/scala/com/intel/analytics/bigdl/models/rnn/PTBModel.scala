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
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{TimeDistributed, _}

object PTBModel {
  def apply(
    inputSize: Int,
    hiddenSize: Int,
    outputSize: Int,
    numLayers: Int,
    interNum: Int = 0)
  : Module[Float] = {
    val input = Input[Float]()
    val embeddingLookup =
      LookupTable[Float](inputSize, hiddenSize).inputs(input)
//    val transpose = Transpose[Float](Array((1, 2))).inputs(embeddingLookup)
    val lstm = addLayer(hiddenSize, hiddenSize, 1, numLayers, embeddingLookup)
    val output = if (interNum > 0) {
//    val linear = Sequential[Float]().
//      add(Linear[Float](hiddenSize, 100)).add(Linear[Float](100, outputSize))
      val linear = Sequential[Float]().add(Linear[Float](hiddenSize, interNum)).add(Linear[Float](interNum, outputSize))
      TimeDistributed[Float](linear).inputs(lstm)
    } else {
      val linear = Linear[Float](hiddenSize, outputSize)
      TimeDistributed[Float](linear).inputs(lstm)
    }
    Graph(input, output)
  }

  def addLayer(inputSize: Int,
               hiddenSize: Int,
               depth: Int,
               numLayers: Int,
               input: ModuleNode[Float]): ModuleNode[Float] = {
    if (depth == numLayers) {
      Recurrent[Float]()
        .add(LSTM[Float](inputSize, hiddenSize, 0))
        .inputs(input)
    } else {
      addLayer(
        inputSize,
        hiddenSize,
        depth + 1,
        numLayers,
        Recurrent[Float]()
          .add(LSTM[Float](inputSize, hiddenSize, 0))
          .inputs(input)
      )
    }
  }
}
