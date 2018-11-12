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
package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.{DataType, Memory, MklDnn}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

class ReorderMemory(inputFormat: MemoryData, outputFormat: MemoryData,
  gradInputFormat: MemoryData, gradOutputFormat: MemoryData
) extends MklDnnLayer {

  _outputFormats = Array(outputFormat)
  _gradInputFormats = Array(gradInputFormat)

  if (this.getName() == "test") {
    val tmp = 0
  }

  override private[bigdl] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = if (inputFormat == null) inputs else Array(inputFormat)

    if (outputFormat != null) {
      _inputFormats = Array(HeapData(_outputFormats(0).shape, _inputFormats(0).layout))
    } else {
      _outputFormats = _inputFormats
    }
    require(_inputFormats.length == 1, "Only accept one tensor as input")

    if (this.getName() == "test") {
      val tmp = 0
    }
    require(_inputFormats(0).shape.product == _outputFormats(0).shape.product,
      "input output memory not match")
    val fwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(
      _inputFormats(0).getPrimitiveDescription(runtime),
      _outputFormats(0).getPrimitiveDescription(runtime))
    val fwdReorderPrim = MklDnn.PrimitiveCreate2(fwdReorderPrimDesc,
      Array(_inputFormats(0).getPrimitive(runtime)), Array(0), 1,
      Array(_outputFormats(0).getPrimitive(runtime)), 1)

    updateOutputPrimitives = Array(fwdReorderPrim)
    output = initTensor(_outputFormats(0))
    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (_inputFormats(0).layout == _outputFormats(0).layout) {
      output = input
    } else {
      output = super.updateOutput(input)
    }
    output
  }

  override private[bigdl] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    _gradInputFormats = (gradInputFormat, inputFormat) match {
      case (null, null) => inputFormats()
      case (null, x) => Array(x)
      case (x, _) => Array(x)
    }

    _gradOutputFormats = if (gradOutputFormat == null) grads else Array(gradOutputFormat)
    _gradOutputFormatsForWeight = if (gradOutputFormat == null) grads else Array(gradOutputFormat)

    if (_gradInputFormats != null) {
      _gradOutputFormats = Array(HeapData(_gradInputFormats(0).shape, _gradOutputFormats(0).layout))
      _gradOutputFormatsForWeight =
        Array(HeapData(_gradInputFormats(0).shape, _gradOutputFormatsForWeight(0).layout))
    }

    if (this.getName() == "test") {
      val tmp = 0
    }
    require(_gradOutputFormats.length == 1, "Only accept one tensor as input")

    require(_gradOutputFormats(0).shape.product == _gradInputFormats(0).shape.product,
      "input output memory not match")
    val bwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(
      _gradOutputFormats(0).getPrimitiveDescription(runtime),
      _gradInputFormats(0).getPrimitiveDescription(runtime))
    val bwdReorderPrim = MklDnn.PrimitiveCreate2(bwdReorderPrimDesc,
      _gradOutputFormats.map(_.getPrimitive(runtime)), Array(0), 1,
      _gradInputFormats.map(_.getPrimitive(runtime)), 1)

    updateGradInputPrimitives = Array(bwdReorderPrim)
    gradInput = initTensor(_gradInputFormats(0))
    (_gradOutputFormats, _gradInputFormats)
  }

  override def toString(): String = {
    if (_inputFormats != null) {
      s"nn.mkl.ReorderMemory(${_inputFormats(0)} -> ${outputFormat})"
    } else {
      s"nn.mkl.ReorderMemory(_ -> ${outputFormat})"
    }
  }
}

object ReorderMemory {
  def apply(inputFormat: MemoryData, outputFormat: MemoryData, gradInputFormat: MemoryData,
    gradOutputFomat: MemoryData): ReorderMemory = {
    new ReorderMemory(inputFormat, outputFormat, gradInputFormat, gradOutputFomat)
  }

  def apply(outputFormat: MemoryData, gradInputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(null, outputFormat, gradInputFormat, null)
  }

  def apply(outputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(null, outputFormat, null, null)
  }
}
