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
    val hiddenSize = 8
    val numHeads = 4
    val filterSize = 4
    val num_hidden_layers = 6 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val block = transformer.block(num_hidden_layers)

    val input1 = Tensor[Float](2, 3, 8).rand()
    val input2 = Tensor[Float](2, 4, 3, 3).rand()

    val out1 = block.forward(T(input1, input2))

    println("done")

  }

  "transformer" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 2 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    // input should be (batchsize, length)
    val input = Tensor[Float](2, 6).fill(1.0f)
    val block = transformer.forward(input)

    println("done")
  }

  "tranformer block" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 1
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val params = transformer.blockModel.getParametersTable()

    val input = Tensor[Float](2, 6).fill(1.0f)
    val block = transformer.forward(input)

    println("done")
  }

  "transformer prepare decode" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 1
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val input = Tensor[Float](T(T(T( 16.24345364,  -6.11756414,  -5.28171752, -10.72968622),
      T(  8.65407629, -23.01538697,  17.44811764,  -7.61206901),
      T(  3.19039096,  -2.49370375,  14.62107937, -20.60140709),
      T( -3.22417204,  -3.84054355,  11.33769442, -10.99891267),
      T( -1.72428208,  -8.77858418,   0.42213747,   5.82815214),
      T(-11.00619177,  11.4472371,    9.01590721,   5.02494339)),
      T(T(  9.00855949,  -6.83727859,  -1.22890226,  -9.35769434),
        T( -2.6788808,    5.30355467,  -6.91660752,  -3.96753527),
        T( -6.871727,    -8.45205641,  -6.71246131,  -0.12664599),
        T(-11.17310349,   2.34415698,  16.59802177,   7.42044161),
        T( -1.91835552,  -8.87628964,  -7.47158294,  16.92454601),
        T(  0.50807755,  -6.36995647,   1.90915485,  21.00255136))))

    val out = transformer.transformerPrepareDecoder(input)

    println("done")
  }
}
