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

class MaskHead(
  val inChannels: Int = 0,
  val resolution: Int = 0,
  val scales: Array[Float],
  val samplingRratio: Float = 0.1f,
  val layers: Array[Int],
  val dilation: Int = 1,
  val numClasses: Int = 81, // coco dataset class number
  val useGn: Boolean = false)(implicit ev: TensorNumeric[Float])
  extends AbstractModule[Table, Table, Float] {

  private val featureExtractor = new MaskRCNNFPNFeatureExtractor(
    inChannels, resolution, scales, samplingRratio, layers, dilation, useGn)
  val dimReduced = layers(layers.length - 1)
  private val predictor = new MaskRCNNC4Predictor(dimReduced, numClasses, dimReduced)
  private val postProcessor = new MaskPostProcessor()

  /**
   * @param input feature-maps from possibly several levels and proposal boxes
   * @return
   * first tensor: the result of the feature extractor
   * second tensor: proposals (list[BoxList]): during training, the original proposals
   *      are returned. During testing, the predicted boxlists are returned
   *      with the `mask` field set
   */
  override def updateOutput(input: Table): Table = {
    val features = input[Table](1)
    val proposals = input[Tensor[Float]](2)
    val imageInfo = input[Tensor[Float]](3)
    val labels = input[Tensor[Float]](4)

    val x = featureExtractor.forward(T(features, proposals))
    val maskLogits = predictor.forward(x)
    val result = postProcessor.forward(T(maskLogits, proposals, imageInfo, labels))
    output = T(x, result)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    throw new UnsupportedOperationException("MaskHead only support inference")
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    val p1 = featureExtractor.parameters()
    val p2 = predictor.parameters()
    (p1._1 ++ p2._1, p1._2 ++ p2._2)
  }
}
