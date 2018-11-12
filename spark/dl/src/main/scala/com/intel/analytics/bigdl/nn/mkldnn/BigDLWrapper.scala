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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import spire.syntax.module


class BigDL2DnnWrapper(val module: AbstractModule[Tensor[_], Tensor[_], Float], formats : String)
  extends MklDnnLayer {

  output = Tensor[Float]()
  gradInput = Tensor[Float]()

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    // todo: only support tensor model and implement computeOutputShape
    val inputShape = if (inputs(0).layout == Memory.Format.nchw ||
    inputs(0).layout == Memory.Format.nChw8c || inputs(0).layout == Memory.Format.nChw16c) {
      inputs(0).shape
    } else {
      val s = inputs(0).shape
      // from nhwc -> nchw
      Array(s(0), s(3), s(1), s(2))
    }
    val outputShape = module.computeOutputShape(Shape(inputShape)).toSingle().toArray

    require(inputShape.length == 2 || inputShape.length == 4,
      s"just support input shape dim is 2 or 4, but get ${inputShape.length}")
    require(outputShape.length == 2 || outputShape.length == 4,
      s"just support output shape dim is 2 or 4, but get ${outputShape.length}")

    val realInputs = if (inputShape.length == 4) {
      HeapData(inputShape, Memory.Format.nchw)
    } else {
      HeapData(inputShape, Memory.Format.nc)
    }

    val realOutputs = if (outputShape.length == 4) {
      HeapData(outputShape, Memory.Format.nchw)
    } else {
      HeapData(outputShape, Memory.Format.nc)
    }

    _inputFormats = Array(realInputs)
    _outputFormats = Array(realOutputs)

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = _outputFormats
    _gradOutputFormatsForWeight = _outputFormats
    _gradInputFormats = _inputFormats
    (_outputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    output = module.forward(input.toTensor[Float])
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = module.updateGradInput(input.toTensor[Float], gradOutput.toTensor[Float])
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    module.accGradParameters(input.toTensor[Float], gradOutput.toTensor[Float])
  }
}


object BigDL2DnnWrapper {
  def apply(module: AbstractModule[Tensor[_], Tensor[_], Float], formats : String)
  : BigDL2DnnWrapper = new BigDL2DnnWrapper(module, formats)
}