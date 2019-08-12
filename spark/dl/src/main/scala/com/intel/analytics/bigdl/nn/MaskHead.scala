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
import com.intel.analytics.bigdl.utils.{T, Table}

class ROIMaskHead()
 (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float] {

  val in_channels: Int = 0
  val resolution: Int = 0
  val scales: Array[Float] = new Array[Float](1)
  val sampling_ratio: Float = 0.1f
  val layers: Array[Int] = new Array[Int](1)
  val dilation: Int = 1
  val use_gn: Boolean = false

  val feature_extractor = new MaskRCNNFPNFeatureExtractor(
    in_channels, resolution, scales, sampling_ratio, layers, dilation, use_gn)

  val num_classes: Int = 0
  val dim_reduced: Int = 0
  val predictor = new MaskRCNNC4Predictor(in_channels, num_classes, dim_reduced)

  val post_processor = new MaskPostProcessor()

  /**
    *         """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
        """
    * @param input
    * @return
    */
  override def updateOutput(input: Table): Table = {
    val features = input[Tensor[Float]](1)
    val proposals = input[Tensor[Float]](2)

    val x = feature_extractor.forward(T(features, proposals))
    val mask_logits = predictor.forward(x)

    val result = post_processor.forward(T(mask_logits, proposals))
    output = T(x, result)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput
  }
}
