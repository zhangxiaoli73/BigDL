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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.dmg.pmml.{False, True}

class MaskRCNNFPNFeatureExtractor(inChannels: Int, resolution: Int,
  scales: Array[Float], samplingRatio: Float, layers: Array[Int],
  dilation: Int, useGn: Boolean = false)
  (implicit ev: TensorNumeric[Float]) extends BaseModule[Float] {

  require(dilation == 1, s"Only support dilation = 1, but get ${dilation}")

  override def buildModel(): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Pooler(resolution, scales, samplingRatio.toInt))

    var nextFeatures = inChannels
    var i = 0
    while (i < layers.length) {
      val features = layers(i)
      // todo: not support dilation convolution
      val module = SpatialConvolution[Float](
        nextFeatures,
        features,
        kernelW = 3,
        kernelH = 3,
        strideW = 1,
        strideH = 1,
        padW = dilation,
        padH = dilation,
        withBias = if (useGn) false else true
      ).setName(s"mask_fcn{${i}}")

      // weight init
      module.setInitMethod(MsraFiller(false), Zeros)
      model.add(module)
      nextFeatures = features
      i += 1
    }
    model.add(ReLU[Float]())
  }
}
