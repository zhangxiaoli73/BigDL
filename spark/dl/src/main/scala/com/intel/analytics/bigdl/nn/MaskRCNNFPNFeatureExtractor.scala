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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.dmg.pmml.{False, True}

class MaskRCNNFPNFeatureExtractor(in_channels: Int, resolution: Int,
  scales: Array[Float], sampling_ratio: Float, layers: Array[Int],
  dilation: Int, use_gn: Boolean = false)
  (implicit ev: TensorNumeric[Float]) extends BaseModule[Float] {

  override def buildModel(): Module[Float] = {
    val model = Sequential[Float]()
    //  val pooler = Pooler((resolution, resolution), scales, sampling_ratio)

    var next_features = in_channels
    var i = 0
    while (i < layers.length) {
      val layer_features = layers(i)
      // todo: not support dilation convolution
      val module = SpatialConvolution[Float](
        next_features,
        layer_features,
        kernelW = 3,
        kernelH = 3,
        strideW = 1,
        strideH = 1,
        padW = dilation,
        padH = dilation,
        withBias = if (use_gn) false else true
      ).setName(s"mask_fcn{${i}}")

      // weight init
      // todo: nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
      module.bias.fill(0.0f)

      model.add(module)
      next_features = layer_features
      i += 1
    }
    model.add(ReLU[Float]())
  }
}
