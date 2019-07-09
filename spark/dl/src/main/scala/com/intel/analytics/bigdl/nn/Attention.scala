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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.language.existentials
import scala.reflect.ClassTag

/**
  * Implementation of multiheaded attention and self-attention layers.
  *
  * @param hiddenSize hidden size
  * @param numHeads heads number
  * @param attentionDropout
  */
class AttentionCache[T: ClassTag](
                                   val hiddenSize: Int, val numHeads: Int, val attentionDropout: Float)
                                 (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] {

  // for prediction
  private val join1 = nn.JoinTable[T](dimension = 2, nInputDims = -1)
  private val join2 = nn.JoinTable[T](dimension = 2, nInputDims = -1)
  private var graph: Graph[T] = null

  private val queryLayer = TransformerOperation.dense(
    hiddenSize, hiddenSize, false, name = s"${this.getName()}_q")
  private val keyLayer = TransformerOperation.dense(
    hiddenSize, hiddenSize, false, name = s"${this.getName()}_k")
  private val valueLayer = TransformerOperation.dense(
    hiddenSize, hiddenSize, false, name = s"${this.getName()}_v")

  private[bigdl] val model : Module[T] = {
    // InputX with shape (batch_size, length_x, hidden_size).
    // InputY with shape (batch_size, length_x, hidden_size)
    // for self attention, InputX and InputY should be the same.
    // Bias is attention bias that will be added to the result of the dot product.
    val inputX = Input()
    val inputY = Input()
    val inputBias = Input()

    val querySplit = new SplitHeads(hiddenSize, numHeads, true)
      .inputs(queryLayer.inputs(inputX))
    val keySplit = new SplitHeads(hiddenSize, numHeads)
      .inputs(keyLayer.inputs(inputY))
    val valueSplit = new SplitHeads(hiddenSize, numHeads)
      .inputs(valueLayer.inputs(inputY))

    val inputQ = Input()
    val inputK = Input()
    val inputV = Input()

    val contiguousQ = new Contiguous[T]().inputs(inputQ)
    val contiguousK = new Contiguous[T]().inputs(inputK)
    val contiguousV = new Contiguous[T]().inputs(inputV)

    val matmul = MM(transB = true).inputs(contiguousQ, contiguousK)
    val cadd = CAddTable().inputs(matmul, inputBias)
    val softMax = TransformerOperation.softMax[T]().inputs(cadd)

    val drop = Dropout(initP = (1.0 - attentionDropout)).inputs(softMax)
    val matmulNoTrans = MM().inputs(drop, contiguousV)
    // Recombine heads --> (batch_size, length, hidden_size)
    val combineHeads = new CombineHeads().inputs(matmulNoTrans)
    // Run the combined outputs through another linear projection layer.
    val outputLayer = TransformerOperation.dense(
      hiddenSize, hiddenSize, false, name = s"${this.getName()}_output_transform")
      .inputs(combineHeads)
    graph = Graph(Array(inputQ, inputK, inputV, inputBias), Array(outputLayer))
    graph.cloneModule()
    //
    //    val m = Graph(Array(inputX, inputY, inputBias),
    //      Array(graph.inputs(querySplit, keySplit, valueSplit, inputBias)))
    //    m
  }

  private def updateOutput(input: Activity): Activity = {
    //    require(input.toTable.length() == 4 && !this.isTraining(),
    //      "Only support 4 inputs for model inference")
    if (input.toTable.length() == 3) {

    }
    val inputTable = input.toTable
    val inputX = inputTable[Tensor[T]](1)
    val inputY = inputTable[Tensor[T]](2)
    val inputBias = inputTable[Tensor[T]](3)
    /**
      * cache: (Used during prediction) dictionary with tensors containing results of
      * previous attentions. The dictionary must have the items:
      * {"k": tensor with shape [batch_size, i, key_channels],
      * "v": tensor with shape [batch_size, i, value_channels]}
      * where i is the current decoded length.
      */
    val cache = inputTable[Table](4)

    val query = queryLayer.forward(inputX).toTensor[T]

    val (inputK, inputV) = if (cache.length() > 0) {
      (cache.apply[Tensor[T]](this.getName() + "_k"),
        cache.apply[Tensor[T]](this.getName() + "_v"))
    } else (null, null)

    val key = if (inputK != null) {
      join1.forward(T(keyLayer.forward(inputY).toTensor[T], inputK))
    } else keyLayer.forward(inputY).toTensor[T]
    val value = if (inputV != null) {
      join2.forward(T(valueLayer.forward(inputY).toTensor[T], inputV))
    } else valueLayer.forward(inputY).toTensor[T]

    // update cache
    if (cache.length() > 0) {
      cache.update(this.getName() + "_k", key)
      cache.update(this.getName() + "_v", value)
    }
    output = graph.updateOutput(T(query, key, value, inputBias))
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = model.updateGradInput(input, gradOutput)
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

  override def clearState(): this.type = {
    model.clearState()
    this
  }
}


object AttentionCache {
  def apply[@specialized(Float, Double) T: ClassTag]
  (hiddenSize: Int, numHeads: Int, attentionDropout: Float)
  (implicit ev: TensorNumeric[T]): AttentionCache[T] =
    new AttentionCache(hiddenSize: Int, numHeads: Int, attentionDropout: Float)
}
