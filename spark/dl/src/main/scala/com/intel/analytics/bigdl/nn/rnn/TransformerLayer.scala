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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{Dropout, Sequential}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Transformer main model_fn.
 * features: Map of features to the model. Should contain the following:
 * "inputs": Transformer inputs. [batch_size, input_length, 1, hidden_dim].
 * "targets": Target decoder outputs. [batch_size, decoder_length, 1, hidden_dim]
 * "target_space_id": A scalar int from data_generators.problem.SpaceID.
 * Returns:
 * Final decoder representation. [batch_size, decoder_length, hidden_dim]
 *
 * @param vocabSize
 * @param hiddenSize
 * @param numHeads
 * @param filterSize
 * @param num_hidden_layers
 * @param postprocessDropout
 * @param attentionDropout
 * @param reluDropout
 * @param allow_ffn_pad
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters.
 */
class TransformerLayer[T: ClassTag](
   vocabSize: Int,
   hiddenSize: Int,
   numHeads: Int,
   filterSize: Int,
   num_hidden_layers: Int,
   postprocessDropout: Float,
   attentionDropout: Float,
   reluDropout: Float,
   allow_ffn_pad: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] {

  val embedding_softmax_layer = new EmbeddingSharedWeights(vocabSize, hiddenSize)
  val postDropOut = Dropout(1- postprocessDropout)
  val blockModel = block(num_hidden_layers)

  override def updateOutput(input: Activity): Activity = {
    require(input.isTensor, "only support LM now")
    updateOutputForLM(input.toTensor[T])
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = gradInput

  private def updateOutputForLM(input: Tensor[T]): Tensor[T] = {
    val inputTensor = input.toTensor[T]
    // Calculate attention bias for encoder self-attention and decoder
    // multi-headed attention layers.
    val attention_bias = TransformerOperation.getPaddingBias(inputTensor)
    // embedded_inputs with shape [batch, length]
    val embedded_inputs = embedding_softmax_layer.forward(input)
    val inputs_padding = TransformerOperation.getPadding(input)
    // todo: add pos encoding
    val drop = if (train) {
      postDropOut.forward(embedded_inputs)
    } else embedded_inputs

    // block for encode
    val attention = blockModel.forward(drop)
    attention.toTensor[T]
  }

  def block(num_layers: Int): Module[T] = {
    val model = Sequential[T]()
    var i = 0
    while (i < num_layers) {
      val attention = new AttentionLayer[T](hiddenSize, numHeads, attentionDropout)
      val ffn = new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
      val attentionModel = prePostProcessingWrapper(attention)
      val ffnModel = prePostProcessingWrapper(ffn)
      model.add(attentionModel).add(ffnModel)
      i += 1
    }
    model
  }

  private def prePostProcessingWrapper(layer: Module[T]): Module[T] = {
    val model = Sequential[T]()
    model.add(new LayerNormalization[T](hiddenSize))
      .add(layer)
      .add(Dropout[T](1 - postprocessDropout))
    // todo: x + y
    model
  }

  private def transformerPrepareDecoder(targets: Tensor[T]) : Unit = {
//      decoder_self_attention_bias = (
//        common_attention.attention_bias_lower_triangle(
//          common_layers.shape_list(targets)[1]))
//
//      decoder_input = common_layers.shift_right_3d(targets)
//      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    // return decoder_input, decoder_self_attention_bias
  }
}
