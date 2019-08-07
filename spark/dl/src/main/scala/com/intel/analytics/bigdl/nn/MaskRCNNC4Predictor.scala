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
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

class MaskRCNNC4Predictor(in_channels: Int, num_classes: Int, dim_reduced: Int)
  (implicit ev: TensorNumeric[Float]) extends BaseModule[Float] {

  override def buildModel(): Module[Float] = {
    val conv5_mask = SpatialFullConvolution(in_channels, dim_reduced,
      kW = 2, kH = 2, dW = 2, dH = 2)
    val mask_fcn_logits = SpatialConvolution(nInputPlane = dim_reduced,
      nOutputPlane = num_classes, kernelW = 1, kernelH = 1, strideH = 1, strideW = 1)

    // init parameters
    // init weight & bias, Caffe2 implementation uses MSRAFill,
    // which in fact corresponds to kaiming_normal_ in PyTorch
    // todo: nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
    RNG.setSeed(100)
    conv5_mask.setInitMethod(MsraFiller(false), Zeros)
    mask_fcn_logits.setInitMethod(MsraFiller(false), Zeros)

    val model = Sequential[Float]()
    model.add(conv5_mask).add(ReLU[Float]()).add(mask_fcn_logits)
    model
  }
}
