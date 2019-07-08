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

import breeze.linalg.*
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime._

/**
 * Transformer model from "Attention Is All You Need".
 * The Transformer model consists of an encoder and a decoder, both are stacks
 * of self-attention layers followed by feed-forward layers. This model yields
 * good results on a number of problems, especially in NLP and machine translation.
 * See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
 * description of the model and the results obtained with its early version.
 * @param hiddenSize
 * @param numHeads
 * @param filterSize
 * @param numHiddenlayers
 * @param embeddingDropout
 * @param attentionDropout
 * @param ffnDropout
 * @param paddingValue padding value for word embedding, default 0, which means no padding.
 * @param withShareWeightsLinear whether to add linear that sharing weights with embedding layer.
 * @param transformerType transformer type, support LanguageModel and Translation.
 * @tparam T The numeric type in this module parameters.
 */
class Transformer[T: ClassTag](
   val vocabSize: Int,
   val hiddenSize: Int,
   val numHeads: Int,
   val filterSize: Int,
   val numHiddenlayers: Int,
   val embeddingDropout: Float,
   val attentionDropout: Float,
   val ffnDropout: Float,
   val paddingValue: Double = 0,
   val withShareWeightsLinear: Boolean = false,
   val transformerType: TransformerType = LanguageModel)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] {


  private val embeddingLayer = Sequential[T]().add(
    LookupTable[T](vocabSize, hiddenSize, paddingValue = paddingValue,
      maskZero = true).setName("embedding")).add(MulConstant(math.sqrt(hiddenSize)))
  private val linearSharedWeigths = TimeDistributed(
    new Linear(inputSize = hiddenSize, outputSize = vocabSize, withBias = false))

  private[bigdl] var decoderStack: Module[T] = createDecoder()
  private[bigdl] var encoderStack: Module[T] = createEncoder()
  private[bigdl] var predictModel: Module[T] = null

  private[bigdl] var model : Module[T] = buildModel()

  private def createDecoder(): Module[T] = {
    val decoderInputNode = Input()
    val decoderSelfAttentionBiasNode = Input()
    val encoderOutputNode = Input()
    val encoderAttentionBiasNode = Input()
    Graph(Array(decoderInputNode, decoderSelfAttentionBiasNode,
      encoderOutputNode, encoderAttentionBiasNode),
      Array(block(numHiddenlayers, decoderInputNode, decoderSelfAttentionBiasNode,
        encoderOutputNode, encoderAttentionBiasNode, blockType = "decoder")))
  }

  private def createEncoder(): Module[T] = {
    val encoderInputNode = Input()
    val encoderAttentionBiasNode = Input()
    Graph(Array(encoderInputNode, encoderAttentionBiasNode),
      Array(block(numHiddenlayers, encoderInputNode, encoderAttentionBiasNode,
        blockType = "encoder")))
  }

  private def buildModel(): Module[T] = {
    transformerType match {
      case LanguageModel => buildLM()
      case Translation => buildTranslation()
    }
  }

  private def buildTranslation(): Module[T] = {
    val mask = new PaddingMask()
    // input: int tensor with shape [batch_size, input_length].
    val inputNode = Input()
    // target: int tensor with shape [batch_size, target_length].
    val targetNode = Input()
    val attentionBias = mask.inputs(inputNode)
    val join = JoinTable(1, -1).inputs(inputNode, targetNode)
    val embeddingForTrain = embeddingLayer.inputs(join)
    val split = new SplitTensor(1, 2).inputs(embeddingForTrain)
    val embeddingInput = SelectTable(1).inputs(split)
    val embeddingOutput = SelectTable(2).inputs(split)

    // create encode
    val embeddingNode = Input()
    val paddingNode = Input()
    val encoderGraph = Graph(Array(embeddingNode, paddingNode),
      encode(embeddingNode, paddingNode))

    // create predict model
    val predictNode = Input()
    val attentionMask = mask.inputs(predictNode)
    val embeddingForPredict = embeddingLayer.inputs(predictNode)
    predictModel = Graph(predictNode,
      encoderGraph.inputs(embeddingForPredict, attentionMask))

    // create training model
    val outputNode = decode(embeddingOutput,
      encoderGraph.inputs(embeddingInput, attentionBias), attentionBias)
    Graph(Array(inputNode, targetNode), outputNode)
  }

  private def buildLM(): Module[T] = {
    val inputNode = Input()
    val constantValue = math.sqrt(hiddenSize)
    val embeddingInput = MulConstant(constantValue).inputs(
      LookupTable[T](vocabSize, hiddenSize, paddingValue = paddingValue,
        maskZero = true).setName("embedding").inputs(inputNode))
    val outputNode = decode(embeddingInput)
    Graph(inputNode, outputNode)
  }

  private def updateOutputLM(input: Tensor[T]): Tensor[T] = {
    output = model.forward(input)
    if (withShareWeightsLinear) {
      val embeddingLayer = model.apply("embedding").get
      val embeddingParams = embeddingLayer.getParameters()
      val linearParams = linearSharedWeigths.getParameters()
      linearParams._1.copy(embeddingParams._1)
      output = linearSharedWeigths.updateOutput(model.output.toTensor[T])
    }
    output.toTensor[T]
  }

  private var rangeBuffer = Tensor[T]()
  private var timeBuffer = Tensor[T]()

  def setSymbols(maxDecodeLength: Int, encoder_outputs: Tensor[T],
    encoder_decoder_attention_bias: Tensor[T], ids: Tensor[T], i: Int): Tensor[T] = {
    val length = maxDecodeLength + 1
    TransformerOperation.initRangeTensor(length, rangeBuffer)
    timeBuffer.resize(length, hiddenSize)
    TransformerOperation.getPositionEncode(length, hiddenSize,
      rangeBuffer = rangeBuffer, outBuffer = timeBuffer)
    val timeSignal = TransformerOperation.getPositionEncode(length, hiddenSize,
      rangeBuffer = rangeBuffer, outBuffer = timeBuffer)

    // size (1, 1, maxDecodeLength, maxDecodeLength)
    val decoderSelfAttentionBias = Tensor[T](1, 1, maxDecodeLength, maxDecodeLength)
    TransformerOperation.attentionBiasLowerTriangle(maxDecodeLength, decoderSelfAttentionBias)

    val idsSize = ids.size()
    val decoder_input = ids.select(2, idsSize(1))
    decoder_input.resize(Array(decoder_input.size(1), 1))
    val decoder_input_embedding = embeddingLayer.forward(decoder_input).toTensor[T].clone()

    val timeSize = timeSignal.size()
    val timingTemp = timeSignal.select(1, i + 1)
    val decoder_input_add = decoder_input_embedding.add(timingTemp)

    val self_attention_bias = decoderSelfAttentionBias.select(3, i + 1)
      .select(3, i + 1).resize(Array(1, 1, 1, i + 1))

    val decoder_outputs = decoderStack.forward(T(decoder_input_add,
      self_attention_bias, encoder_outputs, encoder_decoder_attention_bias)).toTensor[T]

    if (withShareWeightsLinear) {
      val embeddingLayer = model.apply("embedding").get
      val embeddingParams = embeddingLayer.getParameters()
      val linearParams = linearSharedWeigths.getParameters()
      linearParams._1.copy(embeddingParams._1)
    }
    val logits = this.linearSharedWeigths.forward(decoder_outputs)
    logits.squeeze(2)
  }


  private def updateOutputTranslation(input: Activity): Activity = {
    if (input.isTensor) {
      require(!this.isTraining(),
        "Input for Transformer should be tensor when doing translation prediction")
      // inference case
      output = predictModel.forward(input)
    } else {
      require(input.toTable.length() == 2, s"Input should be two tensors when doing " +
        s"translation training, but get ${input.toTable.length()}")
      // training case
      output = model.forward(input)
      if (withShareWeightsLinear) {
        val embeddingLayer = model.apply("embedding").get
        val embeddingParams = embeddingLayer.getParameters()
        val linearParams = linearSharedWeigths.getParameters()
        linearParams._1.copy(embeddingParams._1)
        output = linearSharedWeigths.updateOutput(model.output.toTensor[T])
      }
    }
    output
  }

  override def updateOutput(input: Activity): Activity = {
    if (transformerType == Translation) {
      updateOutputTranslation(input)
    } else {
      updateOutputLM(input.toTensor[T])
    }
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    val grad = if (withShareWeightsLinear) {
      linearSharedWeigths.updateGradInput(model.output.toTensor[T], gradOutput.toTensor[T])
    } else gradOutput
    gradInput = model.updateGradInput(input, grad)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    model.accGradParameters(input, gradOutput)
  }

  override def training(): this.type = {
    train = true
    model.training()
    this
  }

  override def evaluate(): this.type = {
    train = false
    model.evaluate()
    this
  }

  override def getExtraParameter(): Array[Tensor[T]] = {
    model.getExtraParameter()
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    model.getTimes()
  }

  override def resetTimes(): Unit = {
    model.resetTimes()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    model.parameters()
  }

  override def getParametersTable(): Table = {
    model.getParametersTable()
  }

  private[nn] def encode(inputs: ModuleNode[T], attentionBias: ModuleNode[T]): ModuleNode[T] = {
    // Prepare inputs to the layer stack by adding positional encodings and
    // applying dropout.
    val position = new PositionEncode().inputs(inputs)
    val encoderInput = CAddTable().inputs(inputs, position)
    val encoderInputDrop = Dropout(1- embeddingDropout).inputs(encoderInput)

    block(numHiddenlayers, encoderInputDrop, attentionBias, blockType = "encode")
  }

  private[nn] def decode(targets: ModuleNode[T],
                     encoderOutput: ModuleNode[T] = null,
                     attentionBias: ModuleNode[T] = null): ModuleNode[T] = {
    val decoderInput = new PositionEncodeWithShift().inputs(targets)
    val decoderSelfAttentionBias = new SelfAttentionMask().inputs(targets)

    val decoderInputDrop = Dropout(1- embeddingDropout).inputs(decoderInput)
    block(numHiddenlayers, decoderInputDrop,
      decoderSelfAttentionBias, encoderOutput, attentionBias, blockType = "decode")
  }

  private[nn] def block(numLayers: Int,
                        decoderInput: ModuleNode[T],
                        decoderSelfAttentionBias: ModuleNode[T],
                        encoderOutput: ModuleNode[T] = null,
                        encoderAttentionBias: ModuleNode[T] = null,
                        blockType: String): ModuleNode[T] = {

    var input = decoderInput
    var i = 0
    while (i < numLayers) {
      val selfAttention = new Attention[T](hiddenSize, numHeads, attentionDropout)
      val selfAttentionModel = processSelfAttention(
        selfAttention, input, decoderSelfAttentionBias,
        s"${blockType}_self_attention_${i}")
      input = selfAttentionModel

      if (encoderOutput != null && encoderAttentionBias != null) {
        val encdecAttention = new Attention[T](hiddenSize, numHeads, attentionDropout)
        val encdecAttentionModel = processEncDecAttention(
          encdecAttention, input, encoderOutput, encoderAttentionBias,
          s"${blockType}_encdec_attention_${i}")
        input = encdecAttentionModel
      }

      val ffn = new FeedForwardNetwork[T](hiddenSize, filterSize, ffnDropout)
      val ffnModel = processFFN(ffn, input, s"${blockType}_ffn_${i}")
      input = ffnModel

      i += 1
    }
    new LayerNormalization[T](hiddenSize).inputs(input)
  }

  private def processSelfAttention(layer: Module[T], decoderInput: ModuleNode[T],
    decoderSelfAttentionBias: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
        .inputs(decoderInput)
    val drop = Dropout[T](1 - embeddingDropout).setName(preName + "/dropout")
        .inputs(layer.setName(preName + "/self_attention")
        .inputs(norm, norm, decoderSelfAttentionBias))
    CAddTable().inputs(decoderInput, drop)
  }

  private def processEncDecAttention(
    layer: Module[T],
    decoderInput: ModuleNode[T],
    encoderOutput: ModuleNode[T],
    attentionBias: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoderInput)
    val drop = Dropout[T](1 - embeddingDropout).setName(preName + "/dropout")
      .inputs(layer.setName(preName + "/encdec_attention")
        .inputs(norm, encoderOutput, attentionBias))
    CAddTable().inputs(decoderInput, drop)
  }

  private def processFFN(layer: Module[T],
    decoderInput: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoderInput)
    val drop = Dropout[T](1 - embeddingDropout).setName(preName + "/dropout")
      .inputs(layer.setName(preName + "/ffn").inputs(norm))
    CAddTable().inputs(decoderInput, drop)
  }

  override def clearState(): this.type = {
    if (withShareWeightsLinear) linearSharedWeigths.clearState()
    model.clearState()
    this
  }
}

object Transformer extends ModuleSerializable {
  def apply[T: ClassTag](
     vocabSize: Int,
     hiddenSize: Int,
     numHeads: Int,
     filterSize: Int,
     numHiddenlayers: Int,
     embeddingDropout: Float,
     attentionDropout: Float,
     ffnDropout: Float,
     paddingValue: Double = 0,
     withShareWeightsLinear: Boolean = false,
     transformerType: TransformerType = LanguageModel)
   (implicit ev: TensorNumeric[T]): Transformer[T] = {
    new Transformer(vocabSize, hiddenSize, numHeads,
      filterSize, numHiddenlayers,
      embeddingDropout, attentionDropout, ffnDropout, paddingValue,
      withShareWeightsLinear = withShareWeightsLinear, transformerType = transformerType)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap

    val model = DataConverter
      .getAttributeValue(context, attrMap.get("model")).
      asInstanceOf[Module[T]]

    val vocabSize = DataConverter
      .getAttributeValue(context, attrMap.get("vocabSize"))
      .asInstanceOf[Int]

    val hiddenSize = DataConverter
      .getAttributeValue(context, attrMap.get("hiddenSize"))
      .asInstanceOf[Int]

    val numHeads = DataConverter
      .getAttributeValue(context, attrMap.get("numHeads"))
      .asInstanceOf[Int]

    val filterSize = DataConverter
      .getAttributeValue(context, attrMap.get("filterSize"))
      .asInstanceOf[Int]

    val numHiddenlayers = DataConverter
      .getAttributeValue(context, attrMap.get("numHiddenlayers"))
      .asInstanceOf[Int]

    val embeddingDropout = DataConverter
      .getAttributeValue(context, attrMap.get("embeddingDropout"))
      .asInstanceOf[Float]

    val attentionDropout = DataConverter
      .getAttributeValue(context, attrMap.get("attentionDropout"))
      .asInstanceOf[Float]

    val ffnDropout = DataConverter
      .getAttributeValue(context, attrMap.get("ffnDropout"))
      .asInstanceOf[Float]

    val paddingValue = DataConverter
      .getAttributeValue(context, attrMap.get("paddingValue"))
      .asInstanceOf[Double]

    val tag = DataConverter
      .getAttributeValue(context, attrMap.get("transformerType"))
      .asInstanceOf[Int]

    val withShareWeightsLinear = DataConverter
      .getAttributeValue(context, attrMap.get("withShareWeightsLinear"))
      .asInstanceOf[Boolean]

    val transformerType = tag match {
      case 1 => LanguageModel
      case 2 => Translation
      case _ => throw new UnsupportedOperationException(
        s"Only support transformer tag 1 and 2, but get ${tag}")
    }

    val transformer = Transformer(vocabSize, hiddenSize, numHeads, filterSize,
      numHiddenlayers, embeddingDropout, attentionDropout, ffnDropout, paddingValue,
      withShareWeightsLinear = withShareWeightsLinear, transformerType)

    transformer.model = model
    transformer
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
     transformerBuilder : BigDLModule.Builder)(implicit ev: TensorNumeric[T]) : Unit = {

    val transformer = context.moduleData.module.asInstanceOf[Transformer[T]]

    val modelBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, modelBuilder, transformer.model,
      ModuleSerializer.abstractModuleType)
    transformerBuilder.putAttr("model", modelBuilder.build)

    val vocabSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, vocabSizeBuilder,
      transformer.vocabSize, universe.typeOf[Int])
    transformerBuilder.putAttr("vocabSize", vocabSizeBuilder.build)

    val hiddenSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, hiddenSizeBuilder,
      transformer.hiddenSize, universe.typeOf[Int])
    transformerBuilder.putAttr("hiddenSize", hiddenSizeBuilder.build)

    val numHeadsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, numHeadsBuilder,
      transformer.numHeads, universe.typeOf[Int])
    transformerBuilder.putAttr("numHeads", numHeadsBuilder.build)

    val filterSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, filterSizeBuilder,
      transformer.filterSize, universe.typeOf[Int])
    transformerBuilder.putAttr("filterSize", filterSizeBuilder.build)

    val numHiddenlayersBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, numHiddenlayersBuilder,
      transformer.numHiddenlayers, universe.typeOf[Int])
    transformerBuilder.putAttr("numHiddenlayers", numHiddenlayersBuilder.build)

    val embeddingDropoutBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, embeddingDropoutBuilder,
      transformer.embeddingDropout, universe.typeOf[Float])
    transformerBuilder.putAttr("embeddingDropout", embeddingDropoutBuilder.build)

    val attentionDropoutBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, attentionDropoutBuilder,
      transformer.attentionDropout, universe.typeOf[Float])
    transformerBuilder.putAttr("attentionDropout", attentionDropoutBuilder.build)

    val ffnDropoutBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, ffnDropoutBuilder,
      transformer.ffnDropout, universe.typeOf[Float])
    transformerBuilder.putAttr("ffnDropout", embeddingDropoutBuilder.build)

    val paddingValueBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, paddingValueBuilder,
      transformer.paddingValue, universe.typeOf[Double])
    transformerBuilder.putAttr("paddingValue", paddingValueBuilder.build)

    val shareWeightsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, shareWeightsBuilder,
      transformer.withShareWeightsLinear, universe.typeOf[Boolean])
    transformerBuilder.putAttr("withShareWeightsLinear", shareWeightsBuilder.build)

    // for language model, marked as 1
    // for translation model, marked as 2
    val tag = transformer.transformerType match {
      case LanguageModel => 1
      case Translation => 2
      case _ => throw new UnsupportedOperationException(s"Only support LanguageModel" +
        s"and Translation transformer type, but get ${transformer.transformerType}")
    }
    val transformerTypeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, transformerTypeBuilder,
      tag, universe.typeOf[Int])
    transformerBuilder.putAttr("transformerType", transformerTypeBuilder.build)
  }
}

/**
 * Return positional encoding.
 * Calculates the position encoding as a mix of sine and cosine functions with
 * geometrically increasing wavelengths.
 * Defined and formulized in Attention is All You Need, section 3.5.
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
private[nn] class PositionEncode[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  @transient private var rangeBuffer : Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val length = input.size(2)
    val channel = input.size(3)

    if (!output.isEmpty && output.nElement() == length * channel) return output

    if (rangeBuffer == null) rangeBuffer = Tensor[T]()
    TransformerOperation.initRangeTensor(length, rangeBuffer)

    output.resize(length, channel)
    TransformerOperation.getPositionEncode(length, channel,
      rangeBuffer = rangeBuffer, outBuffer = output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isEmpty && gradInput.nElement() == input.nElement()) return gradInput
    gradInput.resizeAs(input).zero()
    gradInput
  }
}

// Return postition encoding with input shift right
private[nn] class PositionEncodeWithShift[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  @transient private var rangeBuffer : Tensor[T] = null
  @transient private var timeBuffer : Tensor[T] = null

  // input a Tensor with shape [batch, length, channels]
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    TransformerOperation.shiftRight3D(input, output)
    val length = output.size(2)
    val channel = output.size(3)

    if (rangeBuffer == null) rangeBuffer = Tensor[T]()
    if (timeBuffer == null) timeBuffer = Tensor[T]()

    if (timeBuffer.nElement() != length * channel) {
      TransformerOperation.initRangeTensor(length, rangeBuffer)
      timeBuffer = Tensor[T]().resize(length, channel)
      TransformerOperation.getPositionEncode(length, channel,
        rangeBuffer = rangeBuffer, outBuffer = timeBuffer)
    }

    val batchSize = input.size(1)
    var i = 1
    while (i <= batchSize) {
      output.select(1, i).add(timeBuffer)
      i += 1
    }
    return output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradInput == null) gradInput = Tensor[T]()
    gradInput.resizeAs(gradOutput).zero()
    val size = gradOutput.size(2)
    var i = 1
    while (i < size) {
      gradInput.select(2, i).copy(gradOutput.select(2, i + 1))
      i += 1
    }
    gradInput
  }
}

/**
 * Calculate bias tensor from padding values in tensor.
 * Bias tensor that is added to the pre-softmax multi-headed attention logits,
 * which has shape [batch_size, num_heads, length, length]. The tensor is zero at
 * non-padding locations, and -1e9 (negative infinity) at padding locations.
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
private[nn] class PaddingMask[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    output = TransformerOperation.getPaddingBias(output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    gradInput
  }
}

// This mask is to hide both <pad> and future words. Used in decode
private[nn] class SelfAttentionMask[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  private val maskValue = -1e9

  /**
   * Create an bias tensor to be added to attention logits.
   * Returns tensor with shape (1, 1, length, length)
   * @param length
   * @tparam T
   * @return
   */
  private def attentionBiasLowerTriangle[T: ClassTag](
    length: Int, output: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val arr = output.storage().array()
    for (i <- 0 to (length - 1)) {
      var j = length - 1
      while (j > i) {
        // reminder: here not 1
        arr(i * length + j) = ev.fromType(maskValue)
        j -= 1
      }
    }
    output.resize(Array(1, 1, length, length))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!output.isEmpty && output.nElement() == input.nElement()) return output
    output.resize(input.size(2), input.size(2)).zero()
    attentionBiasLowerTriangle[T](input.size(2), output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isEmpty && gradInput.nElement() == input.nElement()) return gradInput
    gradInput.resizeAs(input).zero()
    gradInput
  }
}

private[nn] class SplitTensor[T: ClassTag](dimension: Int, num: Int)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Table, T] {

  private val innerLayer = new JoinTable[T](dimension, -1)

  override def updateOutput(input: Tensor[T]): Table = {
    output = T.array(input.split(input.size(dimension) / num, dimension))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    gradInput = innerLayer.forward(gradOutput).toTensor[T]
    gradInput
  }
}
