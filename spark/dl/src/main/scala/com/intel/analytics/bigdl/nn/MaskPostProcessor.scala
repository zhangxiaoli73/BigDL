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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import spire.syntax.field

class MaskPostProcessor()
(implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {

  @transient var rangeBuffer: Tensor[Float] = null
  val sigmoid = Sigmoid[Float]()

  /**
   * @param input feature-maps from possibly several levels, proposal boxes and labels
   * @return the predicted boxlists are returned with the `mask` field set
   */
  override def updateOutput(input: Table): Tensor[Float] = {
    val maskLogits = input[Tensor[Float]](1)
    val bbox = input[Tensor[Float]](2) // N * 4
    val labels = input[Tensor[Float]](3)

    val num_masks = maskLogits.size(1)
    if (rangeBuffer == null || rangeBuffer.nElement() != num_masks) {
      rangeBuffer = Tensor[Float](num_masks)
      TransformerOperation.initRangeTensor(num_masks, rangeBuffer)
    }

    val mask_prob = sigmoid.forward(maskLogits)
    require(labels.nDimension() == 1, s"Labels should be tensor with one dimention," +
      s"but get ${labels.nDimension()}")
    require(rangeBuffer.nElement() == labels.nElement(), s"number of masks should be same" +
      s"with labels, but get ${rangeBuffer.nElement()} ${labels.nElement()}")

    output.resize(rangeBuffer.nElement(), 1, mask_prob.size(3), mask_prob.size(4))

    var i = 1
    while (i <= rangeBuffer.nElement()) {
      val dim = rangeBuffer.valueAt(i).toInt + 1
      val index = labels.valueAt(i).toInt // start from 1
      output.narrow(1, i, 1).copy(mask_prob.narrow(1, i, 1).narrow(2, index, 1))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput
  }
}
