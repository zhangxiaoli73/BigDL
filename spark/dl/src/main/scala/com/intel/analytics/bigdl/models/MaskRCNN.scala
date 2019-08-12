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
package com.intel.analytics.bigdl.models

import breeze.linalg.*
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.{FPN, Sequential}
import com.intel.analytics.bigdl.utils.T

object MaskRCNN {

  def buildBackbone(in_channels: Int, out_channels: Int): Module[Float] = {
    // todo: check resnet50
    val body = ResNet(classNum = 1000, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
      "optnet" -> false, "dataSet" -> DatasetType.ImageNet))

    val in_channels_list = Array(in_channels, in_channels * 2, in_channels * 4, in_channels * 8)
    val fpn = new FPN[Float](in_channels_list, out_channels)

    val model = Sequential[Float]()
    model.add(body).add(fpn)
    return model
  }

  def apply(in_channels: Int, out_channels: Int) : Module[Float] = {
    val backbone = buildBackbone(in_channels, out_channels)
    // val rpn = new RPN(cfg, self.backbone.out_channels)
    // combine roi heads

//    val box_head = BoxHead
//    val mask_head = MaskHead()
//    self.keypoint.feature_extractor = self.box.feature_extractor

    backbone
  }

}
