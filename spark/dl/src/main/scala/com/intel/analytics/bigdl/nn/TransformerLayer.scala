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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

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

  val postDropOut = Dropout(1- postprocessDropout)
  val blockModel = block(num_hidden_layers)

  val model = buildModel()

  override def updateOutput(input: Activity): Activity = {
    require(input.isTensor, "only support LM now")
    output = model.forward(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = model.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    model.accGradParameters(input, gradOutput)
  }

  private def buildModel(): Module[T] = {
    val input = Input()
    val decoder_input = new TransformerPrepareDecoder().inputs(input)
    val decoder_self_attention_bias = new TransformerConstant().inputs(input)

    val decoder_input_lm = if (train) {
      postDropOut.inputs(decoder_input)
    } else decoder_input

    val output = blockModel.inputs(decoder_input_lm, decoder_self_attention_bias)
    Graph(input, output)
  }

  def block(num_layers: Int): Module[T] = {
    val decoder_input = Input()
    val decoder_self_attention_bias = Input()
    var output = decoder_input
//    val ffn = new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
//    val node = ffn.inputs(decoder_input)
//    return Graph(decoder_input, node)

    var i = 0
    while (i < num_layers) {
      val attention = new Attention[T](hiddenSize, numHeads, attentionDropout)
      val ffn = new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
      val attentionModel = prePostProcessingWrapperAttention(
        attention, output, decoder_self_attention_bias, s"attention_${i}")
      val ffnModel = prePostProcessingWrapperFFN(ffn, attentionModel, s"ffn_${i}")
      output = ffnModel
      i += 1
    }
    val norm = new LayerNormalization[T](hiddenSize).setName("norm").inputs(output)
    val model = Graph(Array(decoder_input, decoder_self_attention_bias), Array(norm))
    // val model = Graph(Array(decoder_input), Array(norm))
    model
  }

  private def prePostProcessingWrapperAttention(layer: Module[T], decoder_input: ModuleNode[T],
    decoder_self_attention_bias: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoder_input)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout").inputs(
      layer.setName(preName + "/attention")
        .inputs(norm, decoder_self_attention_bias))
    val add = CAddTable().inputs(decoder_input, drop)
    add
  }
  private def prePostProcessingWrapperFFN(layer: Module[T],
    decoder_input: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoder_input)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout")
      .inputs(layer.setName(preName + "/ffn").inputs(norm))
    val add = CAddTable().inputs(decoder_input, drop)
    add
  }
}

private[nn] class TransformerConstant[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val decoder_self_attention_bias = TransformerOperation.
      attentionBiasLowerTriangle[T](input.size(2))
    output = decoder_self_attention_bias
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    gradInput
  }
}

private[nn] class TransformerPrepareDecoder[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val inputTensor = input.toTensor[T]
    TransformerOperation.shiftRight3D(inputTensor, output)
    val decoder_input_tmp = TransformerOperation.addTimingSignal1D(output)
    output = decoder_input_tmp
    return output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(gradOutput.isTensor, "gradOutput should be tensor")
    if (gradInput == null) gradInput = Tensor[T]()
    val gradInputTensor = gradInput.toTensor[T]
    val gradOutputTensor = gradOutput.toTensor[T]

    gradInputTensor.resizeAs(gradOutput.toTensor[T]).zero()
    val size = gradOutputTensor.size(2)
    var i = 1
    while (i < size) {
      gradInputTensor.select(2, i).copy(gradOutputTensor.select(2, i + 1))
      i += 1
    }
    gradInput
  }
}
