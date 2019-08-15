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
  extends BaseModule[Float] {

  override def buildModel(): Module[Float] = {
    val featureExtractor = this.maskFeatureExtractor(
      inChannels, resolution, scales, samplingRratio, layers, dilation, useGn)
    val dimReduced = layers(layers.length - 1)
    val predictor = this.maskPredictor(dimReduced, numClasses, dimReduced)
    val postProcessor = new MaskPostProcessor()

    /**
     * input: feature-maps from possibly several levels and proposal boxes
     * return:
     * first tensor: the result of the feature extractor
     * second tensor: proposals (list[BoxList]): during training, the original proposals
     *      are returned. During testing, the predicted boxlists are returned
     *      with the `mask` field set
     */
    val features = Input()
    val proposals = Input()
    val labels = Input()

    val maskFeatures = featureExtractor.inputs(features, proposals)
    val maskLogits = predictor.inputs(maskFeatures)
    val result = postProcessor.inputs(maskLogits, proposals, labels)

    Graph(Array(features, proposals, labels), Array(maskFeatures, result))
  }

  private[nn] def maskPredictor(inChannels: Int,
                                numClasses: Int,
                                dimReduced: Int): Module[Float] = {
    val convMask = SpatialFullConvolution(inChannels, dimReduced,
      kW = 2, kH = 2, dW = 2, dH = 2)
    val maskLogits = SpatialConvolution(nInputPlane = dimReduced,
      nOutputPlane = numClasses, kernelW = 1, kernelH = 1, strideH = 1, strideW = 1)

    // init weight & bias, Caffe2 implementation uses MSRAFill,
    convMask.setInitMethod(MsraFiller(false), Zeros)
    maskLogits.setInitMethod(MsraFiller(false), Zeros)

    val model = Sequential[Float]()
    model.add(convMask).add(ReLU[Float]()).add(maskLogits)
    model
  }

  private[nn] def maskFeatureExtractor(inChannels: Int,
                                       resolution: Int,
                                       scales: Array[Float],
                                       samplingRatio: Float,
                                       layers: Array[Int],
                                       dilation: Int,
                                       useGn: Boolean = false): Module[Float] = {

    require(dilation == 1, s"Only support dilation = 1, but get ${dilation}")

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


object MaskHead {
  def apply(inChannels: Int = 0,
  resolution: Int = 0,
  scales: Array[Float],
  samplingRratio: Float = 0.1f,
  layers: Array[Int],
  dilation: Int = 1,
  numClasses: Int = 81, // coco dataset class number
  useGn: Boolean = false)(implicit ev: TensorNumeric[Float]): Module[Float] = {
    new MaskHead(inChannels, resolution, scales, samplingRratio,
      layers, dilation, numClasses, useGn)
  }
}
