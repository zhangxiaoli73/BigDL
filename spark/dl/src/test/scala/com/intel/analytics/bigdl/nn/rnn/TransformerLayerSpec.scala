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

import com.intel.analytics.bigdl.nn.{Graph, Input, Sequential}
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

  "transformer without embedding" should "work correctly" in {
    val vocabSize = 1000
    val hiddenSize = 512
    val numHeads = 8
    val filterSize = 2048
    val num_hidden_layers = 2 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 0.1f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    // val input = Tensor[Float](10, 35, 650).fill(1.0f)
    val input = Tensor[Float](10, 35, hiddenSize).rand()
    val block = transformer.forward(input)
    println(block)

    val res = transformer.backward(input, block)

    println("done")
  }

  "transformer without embedding with multi block 1111" should "work correctly" in {
    val vocabSize = 1000
    val hiddenSize = 512
    val numHeads = 8
    val filterSize = 2048
    val num_hidden_layers = 2 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 0.1f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val layer = transformer.blockModel

    val input1 = Tensor[Float](10, 35, hiddenSize).rand()
    val input2 = Tensor[Float](1, 1, 35, 35).rand()
    val gradOutput = Tensor[Float](10, 35, hiddenSize).rand()

    val block = layer.forward(T(input1, input2))
    println(block)

    val res = layer.backward(T(input1, input2), gradOutput)
    println("done")
  }

  "transformer without embedding with multi block" should "work correctly" in {
    val vocabSize = 1000
    val hiddenSize = 512
    val numHeads = 8
    val filterSize = 2048
    val num_hidden_layers = 1 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 0.1f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val layer = transformer.block2()
    val layer2 = transformer.block(1)

    val model = transformer.block(2) // Sequential[Float]().add(layer).add(layer2)

    val input1 = Tensor[Float](10, 35, hiddenSize).rand()
    val input2 = Tensor[Float](1, 1, 35, 35).rand()
    val gradOutput = Tensor[Float](10, 35, hiddenSize).rand()

    val block = layer.forward(T(input1, input2))
    val block2 = layer2.forward(T(block, input2))
    val block3 = model.forward(T(input1, input2))
    println(block)

    // val res3 = model.backward(T(input1, input2), gradOutput)
    val res = layer2.backward(T(block, input2), gradOutput)
    val res2 = layer.backward(T(input1, input2), res)
    println("done")
  }

  "Test multi block" should "work correctly" in {
    val vocabSize = 1000
    val hiddenSize = 512
    val numHeads = 8
    val filterSize = 2048
    val num_hidden_layers = 1 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 0.1f
    val reluDropout = 1.0f
    val layer = new Test[Float](
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val input1 = Tensor[Float](10, 35, hiddenSize).rand()
    val input2 = Tensor[Float](1, 1, 35, 35).rand()
    val gradOutput = Tensor[Float](10, 35, hiddenSize).rand()

    val block = layer.forward(T(input1, input2))
    println(block)

    // val res3 = model.backward(T(input1, input2), gradOutput)
    val res = layer.backward(T(input1, input2), gradOutput)
    println("done")
  }

  "transformer without embedding with multi hidden layers" should "work correctly" in {
    val vocabSize = 1000
    val hiddenSize = 512
    val numHeads = 8
    val filterSize = 2048
    val num_hidden_layers = 2 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 0.1f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    // val input = Tensor[Float](10, 35, 650).fill(1.0f)
    val input = Tensor[Float](10, 35, hiddenSize).rand()
    val block = transformer.forward(input)
    println(block)

    val res = transformer.backward(input, block)

    println("done")
  }

  "transformer block without embedding" should "work correctly" in {
    val vocabSize = 1000
    val hiddenSize = 512
    val numHeads = 8
    val filterSize = 2048
    val num_hidden_layers = 1 // number of blocks
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](
      vocabSize, hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val block = transformer.blockModel

    val input = Tensor[Float](10, 35, hiddenSize).rand()
    val bias = Tensor[Float](1, 1, 35, 35).rand()

    val res = block.forward(T(input, bias))

    val out2 = transformer.backward(T(input, bias), res)

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

    val block = transformer.blockModel

    val input = Tensor[Float](2, 6, 4).rand()
    val bias = Tensor[Float](1, 1, 6, 6).rand()

    val res = block.forward(T(input, bias))

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

  "transformer prepare decode layer" should "work correctly" in {
    val prepare = new TransformerPrepareDecoder[Float]()

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

    val expectedOutput = Tensor[Float](
      T(T(T(  0,          0,           1,          1      ),
      T( 17.084925,    -6.117464,    -4.741415,    -9.729686),
      T(  9.563374,   -23.015186,    17.031971,    -6.612069),
      T(  3.331511,    -2.493404,    13.631087,   -19.601408),
      T( -3.9809747,   -3.8401434,   10.684051,    -9.998913),
      T( -2.6832063,   -8.778085,     0.7057997,    6.828152)),
      T(T(  0,           0,           1,           1       ),
      T(  9.85003,     -6.837178,    -0.68859994,  -8.357695),
      T( -1.7695832,    5.3037543,   -7.332754,    -2.9675353),
      T( -6.730607,    -8.4517565,   -7.702454,     0.87335396),
      T(-11.929906,     2.344557,    15.944379,     8.420442),
      T( -2.8772798,   -8.87579,     -7.1879206,   17.924545))))

    val expectedGradInput = Tensor[Float](
      T(T(T( 17.084925,    -6.117464,    -4.741415,    -9.729686),
        T(  9.563374,   -23.015186,    17.031971,    -6.612069),
        T(  3.331511,    -2.493404,    13.631087,   -19.601408),
        T( -3.9809747,   -3.8401434,   10.684051,    -9.998913),
        T( -2.6832063,   -8.778085,     0.7057997,    6.828152),
        T(  0,           0,           0,          0    )),
      T(T(  9.85003,     -6.837178,    -0.68859994,  -8.357695),
        T( -1.7695832,    5.3037543,   -7.332754,    -2.9675353),
        T( -6.730607,    -8.4517565,   -7.702454,     0.87335396),
        T(-11.929906,     2.344557,    15.944379,     8.420442),
        T( -2.8772798,   -8.87579,     -7.1879206,   17.924545),
        T(  0,           0,           0,           0))))

    val out = prepare.forward(input)
    out should be(expectedOutput)

    val out2 = prepare.backward(input, out)
    out2 should be(expectedGradInput)

  }

  "transformer constant layer" should "work correctly" in {
    val prepare = new TransformerConstant[Float]()

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

    val expectedOutput = Tensor[Float](
      T(T(T(T(0.0f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, -1e9f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)))))

    val out = prepare.forward(input)
    out should be(expectedOutput)

    val out2 = prepare.backward(input, out)
    println("done")
  }

  "transformer prepare layer" should "work correctly" in {
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

    val expectedOutput = Tensor[Float](
      T(T(T(T(0.0f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f),
        T(0.0f, 0.0f, -1e9f, -1e9f, -1e9f, -1e9f),
        T(0.0f, 0.0f, 0.0f, -1e9f, -1e9f, -1e9f),
        T(0.0f, 0.0f, 0.0f, 0.0f, -1e9f, -1e9f),
        T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1e9f),
        T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)))))


    val inputNode = Input[Float]()
    val decoder_input = new TransformerPrepareDecoder[Float]().inputs(inputNode)
    val decoder_self_attention_bias = new TransformerConstant[Float]().inputs(inputNode)

    val prepare = Graph(Array(inputNode), Array(decoder_input, decoder_self_attention_bias))

    val out = prepare.forward(input)
    // out should be(expectedOutput)

    val out2 = prepare.backward(input, out)
    println("done")
  }
}
