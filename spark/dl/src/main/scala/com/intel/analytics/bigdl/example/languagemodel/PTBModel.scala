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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.rnn.TransformerLayer
import com.intel.analytics.bigdl.nn.{TimeDistributed, _}

object PTBModel {
  def applyTransformer(
             inputSize: Int,
             hiddenSize: Int,
             outputSize: Int,
             numLayers: Int,
             keepProb: Float = 2.0f)
  : Module[Float] = {
    val input = Input[Float]()
    val hiddenSize_tmp = 512
    val embeddingLookup = LookupTable[Float](inputSize, hiddenSize_tmp).inputs(input)
    val transform = new TransformerLayer[Float](vocabSize = 1000,
      hiddenSize = hiddenSize_tmp, numHeads = 8, filterSize = 2048,
      num_hidden_layers = 1, postprocessDropout = 1.0f,
      attentionDropout = 0.1f, reluDropout = 0.1f).inputs(embeddingLookup)
    val linear = Linear[Float](hiddenSize, outputSize)
    val output = TimeDistributed[Float](linear).inputs(transform)
    Graph(input, output)
  }

  def apply(
    inputSize: Int,
    hiddenSize: Int,
    outputSize: Int,
    numLayers: Int,
    keepProb: Float = 2.0f)
  : Module[Float] = {
    val input = Input[Float]()
    val embeddingLookup = LookupTable[Float](inputSize, hiddenSize).inputs(input)

    val inputs = if (keepProb < 1) {
      Dropout[Float](keepProb).inputs(embeddingLookup)
    } else embeddingLookup

    val lstm = addLayer(hiddenSize, hiddenSize, 1, numLayers, inputs)
    val linear = Linear[Float](hiddenSize, outputSize)
    val output = TimeDistributed[Float](linear).inputs(lstm)

    Graph(input, output)
  }

  private def addLayer(inputSize: Int,
               hiddenSize: Int,
               depth: Int,
               numLayers: Int,
               input: ModuleNode[Float]): ModuleNode[Float] = {
    if (depth == numLayers) {
      Recurrent[Float]()
        .add(LSTM[Float](inputSize, hiddenSize, 0, null, null, null))
        .inputs(input)
    } else {
      addLayer(
        inputSize,
        hiddenSize,
        depth + 1,
        numLayers,
        Recurrent[Float]()
          .add(LSTM[Float](inputSize, hiddenSize, 0, null, null, null))
          .inputs(input)
      )
    }
  }
}
