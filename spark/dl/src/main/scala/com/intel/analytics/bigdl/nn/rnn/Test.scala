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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Dropout, Graph, Input}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

class Test[T: ClassTag](
  hiddenSize: Int,
  numHeads: Int,
  filterSize: Int,
  num_hidden_layers: Int,
  postprocessDropout: Float,
  attentionDropout: Float,
  reluDropout: Float)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Activity, T] {

  val attentionLayer = new AttentionLayer[T](hiddenSize, numHeads, attentionDropout)
  val ffnLayer = new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
  val normLayer = new LayerNormalization[T](hiddenSize)

  val attentionLayer2 = new AttentionLayer[T](hiddenSize, numHeads, attentionDropout)
  val ffnLayer2 = new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
  val normLayer2 = new LayerNormalization[T](hiddenSize)

  val normLayer3 = new LayerNormalization[T](hiddenSize)


  private var norm1 : Activity = null
  private var layer1 : Activity = null
  private var norm2 : Activity = null
  private var ffn1 : Activity = null
  private var norm3 : Activity = null
  private var layer2 : Activity = null
  private var norm4 : Activity = null
  private var ffn2 : Activity = null

  override def updateOutput(input: Activity): Activity = {
    val decoder_input = input.toTable.apply[Tensor[T]](1)
    val decoder_self_attention_bias = input.toTable.apply[Tensor[T]](2)

    // round 1
    norm1 = normLayer.forward(decoder_input)
    layer1 = attentionLayer.forward(T(norm1, decoder_self_attention_bias))
    norm2 = normLayer.forward(layer1)
    ffn1 = ffnLayer.forward(norm2)

    // round 2
    norm3 = normLayer2.forward(ffn1)
    layer2 = attentionLayer2.forward(T(norm3, decoder_self_attention_bias))
    norm4 = normLayer2.forward(layer2)
    ffn2 = ffnLayer2.forward(norm4)

    output = normLayer3.forward(ffn2.toTensor[T])
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    val decoder_input = input.toTable.apply[Tensor[T]](1)
    val decoder_self_attention_bias = input.toTable.apply[Tensor[T]](2)
    val grad = gradOutput.toTensor[T]

    val grad1 = normLayer3.backward(ffn2, grad)

    // round 2
    val ffn2_grad = ffnLayer2.backward(norm4, grad1)
    val norm4_grad = normLayer2.backward(layer2, ffn2_grad)
    val layer2_grad = attentionLayer2.backward(T(norm3, decoder_self_attention_bias), norm4_grad)
    val norm3_grad = normLayer2.backward(ffn1, layer2_grad.toTable.apply[Tensor[T]](1))

    // round 1
    val ffn1_grad = ffnLayer.backward(norm2, norm3_grad)
    val norm2_grad = normLayer.backward(layer1, ffn1_grad)
    val layer1_grad = attentionLayer.backward(T(norm1, decoder_self_attention_bias), norm2_grad)
    val norm1_grad = normLayer.backward(decoder_input, layer1_grad.toTable.apply[Tensor[T]](1))

    gradInput = norm1_grad
    gradInput
  }
}
