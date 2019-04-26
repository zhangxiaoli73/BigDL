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
package com.intel.analytics.bigdl.nn.rnn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class TransformerLayerSpec extends FlatSpec with Matchers {

  "block" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 3
    val numHeads = 2
    val filterSize = 4
    val num_hidden_layers = 6 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer(
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val block = transformer.block(num_hidden_layers)

    val input1 = Tensor[Float]()
    val input2 = input1.clone()
    val input3 = Tensor[Float]()

    val out1 = block.forward(T(input1, input2, input3))

    println("done")

  }

  "transformer" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 3
    val numHeads = 2
    val filterSize = 4
    val num_hidden_layers = 6 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer(
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val input = Tensor[Float]()

    val out1 = transformer.forward(input)

    println("done")
  }
}
