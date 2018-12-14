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

package com.intel.analytics.bigdl.nn.mkldnn.models

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.nn.{SpatialConvolution => _, _}
import com.intel.analytics.bigdl.{Module, nn}

object Vgg_16 {
  def apply(batchSize: Int, classNum: Int, hasDropout: Boolean = true): mkldnn.Sequential = {
    import com.intel.analytics.bigdl.nn.mkldnn._
    val model = mkldnn.Sequential()
    model.add(mkldnn.Input(Array(batchSize, 3, 224, 224), Memory.Format.nchw))
    model.add(Conv(3, 64, 3, 3, 1, 1, 1, 1).setName("conv1_1"))
    model.add(mkldnn.ReLU().setName("relu1_1"))
    model.add(Conv(64, 64, 3, 3, 1, 1, 1, 1).setName("conv1_2"))
    model.add(mkldnn.ReLU().setName("relu1_2"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool1"))

    model.add(Conv(64, 128, 3, 3, 1, 1, 1, 1).setName("conv2_1"))
    model.add(mkldnn.ReLU().setName("relu2_1"))
    model.add(Conv(128, 128, 3, 3, 1, 1, 1, 1).setName("conv2_2"))
    model.add(mkldnn.ReLU().setName("relu2_2"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool2"))

    model.add(Conv(128, 256, 3, 3, 1, 1, 1, 1).setName("conv3_1"))
    model.add(mkldnn.ReLU().setName("relu3_1"))
    model.add(Conv(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_2"))
    model.add(mkldnn.ReLU().setName("relu3_2"))
    model.add(Conv(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_3"))
    model.add(mkldnn.ReLU().setName("relu3_3"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool3"))

    model.add(Conv(256, 512, 3, 3, 1, 1, 1, 1).setName("conv4_1"))
    model.add(mkldnn.ReLU().setName("relu4_1"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_2"))
    model.add(mkldnn.ReLU().setName("relu4_2"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_3"))
    model.add(mkldnn.ReLU().setName("relu4_3"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool4"))

    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_1"))
    model.add(mkldnn.ReLU().setName("relu5_1"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_2"))
    model.add(mkldnn.ReLU().setName("relu5_2"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_3"))
    model.add(mkldnn.ReLU().setName("relu5_3"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool5"))

    model.add(mkldnn.Linear(512 * 7 * 7, 4096)
      .setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc6"))
    model.add(mkldnn.ReLU().setName("relu6"))
    if (hasDropout) model.add(mkldnn.Dropout(0.5).setName("drop6"))
    model.add(mkldnn.Linear(4096, 4096).setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc7"))
    model.add(mkldnn.ReLU().setName("relu7"))
    if (hasDropout) model.add(mkldnn.Dropout(0.5).setName("drop7"))
    model.add(mkldnn.Linear(4096, classNum).setInitMethod(Xavier, ConstInitMethod(0.1)).setName(("fc8")))
    model.add(ReorderMemory(HeapData(Array(batchSize, classNum), Memory.Format.nc)))

    model
  }

  def blas(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    import com.intel.analytics.bigdl.nn
    val conv1_1 = ConvBlas(3, 64, 3, 3, 1, 1, 1, 1).setName("conv1_1").inputs()
    val relu1_1 = nn.ReLU().setName("relu1_1").inputs(conv1_1)
    val conv1_2 = ConvBlas(64, 64, 3, 3, 1, 1, 1, 1).setName("conv1_2").inputs(relu1_1)
    val relu1_2 = nn.ReLU().setName("relu1_2").inputs(conv1_2)
    val pool1 = nn.SpatialMaxPooling(2, 2, 2, 2).setName("pool1").inputs(relu1_2)

    val conv2_1 = ConvBlas(64, 128, 3, 3, 1, 1, 1, 1).setName("conv2_1").inputs(pool1)
    val relu2_1 = nn.ReLU().setName("relu2_1").inputs(conv2_1)
    val conv2_2 = ConvBlas(128, 128, 3, 3, 1, 1, 1, 1).setName("conv2_2").inputs(relu2_1)
    val relu2_2 = nn.ReLU().setName("relu2_2").inputs(conv2_2)
    val pool2 = nn.SpatialMaxPooling(2, 2, 2, 2).setName("pool2").inputs(relu2_2)

    val conv3_1 = ConvBlas(128, 256, 3, 3, 1, 1, 1, 1).setName("conv3_1").inputs(pool2)
    val relu3_1 = nn.ReLU().setName("relu3_1").inputs(conv3_1)
    val conv3_2 = ConvBlas(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_2").inputs(relu3_1)
    val relu3_2 = nn.ReLU().setName("relu3_2").inputs(conv3_2)
    val conv3_3 = ConvBlas(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_3").inputs(relu3_2)
    val relu3_3 = nn.ReLU().setName("relu3_3").inputs(conv3_3)
    val pool3 = nn.SpatialMaxPooling(2, 2, 2, 2).setName("pool3").inputs(relu3_3)

    val conv4_1 = ConvBlas(256, 512, 3, 3, 1, 1, 1, 1).setName("conv4_1").inputs(pool3)
    val relu4_1 = nn.ReLU().setName("relu4_1").inputs(conv4_1)
    val conv4_2 = ConvBlas(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_2").inputs(relu4_1)
    val relu4_2 = nn.ReLU().setName("relu4_2").inputs(conv4_2)
    val conv4_3 = ConvBlas(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_3").inputs(relu4_2)
    val relu4_3 = nn.ReLU().setName("relu4_3").inputs(conv4_3)
    val pool4 = nn.SpatialMaxPooling (2, 2, 2, 2).setName("pool4").inputs(relu4_3)

    val conv5_1 = ConvBlas(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_1").inputs(pool4)
    val relu5_1 = nn.ReLU().setName("relu5_1").inputs(conv5_1)
    val conv5_2 = ConvBlas(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_2").inputs(relu5_1)
    val relu5_2 = nn.ReLU().setName("relu5_2").inputs(conv5_2)
    val conv5_3 = ConvBlas(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_3").inputs(relu5_2)
    val relu5_3 = nn.ReLU().setName("relu5_3").inputs(conv5_3)
    val pool5 = nn.SpatialMaxPooling(2, 2, 2, 2).setName("pool5").inputs(relu5_3)

    
    val view = nn.View(512 * 7 * 7).inputs(pool5)
    val fc6 = nn.Linear(512 * 7 * 7, 4096).
      setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc6").inputs(view)
    val relu6 = nn.ReLU().setName("relu6").inputs(fc6)
    val drop6 = if (hasDropout) {
      nn.Dropout(0.5).setName("drop6").inputs(relu6)
    } else {
      relu6
    }
    val fc7 = nn.Linear(4096, 4096).
      setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc7").inputs(drop6)
    val relu7 = nn.ReLU().setName("relu7").inputs(fc7)
    val drop7 = if (hasDropout) {
      nn.Dropout(0.5).setName("drop7").inputs(relu7)
    } else {
      relu7
    }
    val fc8 = nn.Linear(4096, classNum).
      setInitMethod(Xavier, ConstInitMethod(0.1)).setName(("fc8")).inputs(drop7)

    new StaticGraph[Float](Array(conv1_1), Array(fc8))
  }

  private object ConvBlas {
    def apply(
               nInputPlane: Int,
               nOutputPlane: Int,
               kernelW: Int,
               kernelH: Int,
               strideW: Int = 1,
               strideH: Int = 1,
               padW: Int = 0,
               padH: Int = 0,
               nGroup: Int = 1,
               propagateBack: Boolean = true): nn.SpatialConvolution[Float] = {
      import com.intel.analytics.bigdl.nn
      val conv = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack)
      conv.setInitMethod(Xavier.setVarianceNormAverage(false), Zeros)
      conv
    }
  }

  private object Conv {
    def apply(
      nInputPlane: Int,
      nOutputPlane: Int,
      kernelW: Int,
      kernelH: Int,
      strideW: Int = 1,
      strideH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      nGroup: Int = 1,
      propagateBack: Boolean = true): SpatialConvolution = {
      val conv = SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack)
      conv.setInitMethod(Xavier.setVarianceNormAverage(false), Zeros)
      conv
    }
  }

  def graph(batchSize: Int, classNum: Int, hasDropout: Boolean = true): DnnGraph = {
    import com.intel.analytics.bigdl.nn.mkldnn._
    val input = Input(Array(batchSize, 3, 224, 224), Memory.Format.nchw).inputs()
    val conv1_1 = Conv(3, 64, 3, 3, 1, 1, 1, 1).setName("conv1_1").inputs(input)
    val relu1_1 = ReLU().setName("relu1_1").inputs(conv1_1)
    val conv1_2 = Conv(64, 64, 3, 3, 1, 1, 1, 1).setName("conv1_2").inputs(relu1_1)
    val relu1_2 = ReLU().setName("relu1_2").inputs(conv1_2)
    val pool1 = MaxPooling(2, 2, 2, 2).setName("pool1").inputs(relu1_2)

    val conv2_1 = Conv(64, 128, 3, 3, 1, 1, 1, 1).setName("conv2_1").inputs(pool1)
    val relu2_1 = ReLU().setName("relu2_1").inputs(conv2_1)
    val conv2_2 = Conv(128, 128, 3, 3, 1, 1, 1, 1).setName("conv2_2").inputs(relu2_1)
    val relu2_2 = ReLU().setName("relu2_2").inputs(conv2_2)
    val pool2 = MaxPooling(2, 2, 2, 2).setName("pool2").inputs(relu2_2)

    val conv3_1 = Conv(128, 256, 3, 3, 1, 1, 1, 1).setName("conv3_1").inputs(pool2)
    val relu3_1 = ReLU().setName("relu3_1").inputs(conv3_1)
    val conv3_2 = Conv(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_2").inputs(relu3_1)
    val relu3_2 = ReLU().setName("relu3_2").inputs(conv3_2)
    val conv3_3 = Conv(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_3").inputs(relu3_2)
    val relu3_3 = ReLU().setName("relu3_3").inputs(conv3_3)
    val pool3 = MaxPooling(2, 2, 2, 2).setName("pool3").inputs(relu3_3)

    val conv4_1 = Conv(256, 512, 3, 3, 1, 1, 1, 1).setName("conv4_1").inputs(pool3)
    val relu4_1 = ReLU().setName("relu4_1").inputs(conv4_1)
    val conv4_2 = Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_2").inputs(relu4_1)
    val relu4_2 = ReLU().setName("relu4_2").inputs(conv4_2)
    val conv4_3 = Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_3").inputs(relu4_2)
    val relu4_3 = ReLU().setName("relu4_3").inputs(conv4_3)
    val pool4 = MaxPooling(2, 2, 2, 2).setName("pool4").inputs(relu4_3)

    val conv5_1 = Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_1").inputs(pool4)
    val relu5_1 = ReLU().setName("relu5_1").inputs(conv5_1)
    val conv5_2 = Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_2").inputs(relu5_1)
    val relu5_2 = ReLU().setName("relu5_2").inputs(conv5_2)
    val conv5_3 = Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_3").inputs(relu5_2)
    val relu5_3 = ReLU().setName("relu5_3").inputs(conv5_3)
    val pool5 = MaxPooling(2, 2, 2, 2).setName("pool5").inputs(relu5_3)

    val fc6 = Linear(512 * 7 * 7, 4096).
      setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc6").inputs(pool5)
    val relu6 = ReLU().setName("relu6").inputs(fc6)
    val drop6 = if (hasDropout) {
      Dropout(0.5).setName("drop6").inputs(relu6)
    } else {
      relu6
    }
    val fc7 = Linear(4096, 4096).
      setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc7").inputs(drop6)
    val relu7 = ReLU().setName("relu7").inputs(fc7)
    val drop7 = if (hasDropout) {
      Dropout(0.5).setName("drop7").inputs(relu7)
    } else {
      relu7
    }
    val fc8 = Linear(4096, classNum).
      setInitMethod(Xavier, ConstInitMethod(0.1)).setName(("fc8")).inputs(drop7)
    val output = ReorderMemory(HeapData(Array(batchSize, classNum), Memory.Format.nc)).inputs(fc8)

    DnnGraph(Array(input), Array(output))
  }
}
