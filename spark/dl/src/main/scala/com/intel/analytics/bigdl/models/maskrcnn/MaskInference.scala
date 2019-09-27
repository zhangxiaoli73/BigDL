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

package com.intel.analytics.bigdl.models.mask

import java.io.File
import java.nio.file.{Path, Paths}

import breeze.linalg.{*, max, min, shuffle, where}
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.bigdl.dataset.segmentation.COCO.{MaskAPI, SegmentationMask}
import com.intel.analytics.bigdl.models.maskrcnn.{Mask, MaskRCNN, MaskUtils}
import com.intel.analytics.bigdl.nn.ResizeBilinear
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelNormalize, ColorJitter, Resize}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.commons.io.FileUtils
import org.dmg.pmml.False
import spire.std.float

import scala.collection.mutable.ArrayBuffer
import scala.reflect.io.Path

object MaskInference {

  def main(args: Array[String]): Unit = {
    val minSize = 800
    val maxSize = 1333

    val path = "/home/zhangli/workspace/tmp/mask/maskrcnn-benchmark/demo/weight/"
    val input = MaskUtils.loadWeight(path + "input.txt", Array(1, 3, 800, 1088))

    val imagePath = "/home/zhangli/workspace/tmp/mask/3915380994_2e611b1779_z.jpg"

    val bytes = FileUtils.readFileToByteArray(new File(imagePath))
    val imagefeature = ImageFeature(bytes)

    // transformer
    val trans = BytesToMat() -> ResizeMask(minSize, maxSize) ->
      ChannelNormalize(123f, 115f, 102.9801f) -> MatToTensor[Float]()

    val out = trans.transform(imagefeature)

    val resNetOutChannels = 256
    val backboneOutChannels = 256
    val mask = new MaskRCNN(resNetOutChannels, backboneOutChannels)


    println("done")

  }

  // box shape: box_number * 4
  // mask shape: box_number * 1* 28 * 28
  def postProcessorForMask(classes: Tensor[Float], bboxes: Tensor[Float], scores: Tensor[Float],
    masks: Array[Tensor[Float]], imageHeight: Int, imageWidth: Int): Table = {
    // convert mask to rle mask
    require(masks.length == bboxes.size(1), s"error get ${masks.length} ${bboxes.size(1)}")
    val boxNumber = masks.length
    val masksRLE = new Array[Tensor[Float]](masks.length)
    var i = 0
    while (i < boxNumber) {
      val binaryMask = Mask.pasteMaskInImage(masks(i), bboxes.select(1, i), imageHeight, imageWidth)
      masksRLE(i) = MaskAPI.binaryToRLE(binaryMask).toRLETensor()
      i += 1
    }

    // prepare for evaluation
    val postOutput = T()
    postOutput.update(RoiLabel.MASKS, masksRLE)
    postOutput.update(RoiLabel.BBOXES, bboxes)
    postOutput.update(RoiLabel.CLASSES, classes)
    postOutput.update(RoiLabel.IMGSIZE, (imageHeight, imageWidth))

    postOutput
  }
}


