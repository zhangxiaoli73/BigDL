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

package com.intel.analytics.bigdl.dataset.image


import org.opencv.core.{CvType, Mat, Rect}
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.opencv.OpenCV
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import spire.math.Interval.Open

import scala.collection.Iterator

object CaffeImgCropper {
  def apply(cropWidth: Int, cropHeight: Int,
            mirror: Boolean, cropperMethod: CropperMethod = CropRandom): CaffeImgCropper =
    new CaffeImgCropper(cropHeight, cropWidth, mirror, cropperMethod)
}

// if training, should be random crop, otherwise center crop
class CaffeImgCropper(cropWidth: Int, cropHeight: Int,
                      mirror: Boolean, cropperMethod: CropperMethod = CropRandom)
  extends Transformer[ImageFeature, LabeledBGRImage] {

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new LabeledBGRImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[ImageFeature]): Iterator[LabeledBGRImage] = prev.map(img => {
    val height = img.opencvMat().size().height.toInt
    val width = img.opencvMat().size().width.toInt

    val (startH, startW) = cropperMethod match {
      case CropRandom =>
        val indexH =
          if (height == cropHeight) 0 else math.ceil(RNG.uniform(1e-2, height - cropHeight)).toInt
        val indexW =
          if (width == cropWidth) 0 else math.ceil(RNG.uniform(1e-2, width - cropWidth + 1)).toInt
        (indexH, indexW)
      case CropCenter =>
        ((height - cropHeight) / 2, (width - cropWidth) / 2)
    }

    val input = img.toTensor(ImageFeature.imageTensor)
    cropper(input.storage().array(), buffer.content,
      Array(height, width), Array(buffer.height(), buffer.width()), startH, startW, mirror)

    buffer.setLabel(img.getLabel)
  })

  // size contains height & width
  def cropper(source: Array[Float], target: Array[Float], srcSize: Array[Int],
              tarSize: Array[Int], startH: Int, startW: Int, mirror: Boolean = false): Unit = {
    val height = srcSize(0)
    val width = srcSize(1)
    val cropHeight = tarSize(0)
    val cropWidth = tarSize(1)

    val startIndex = startW + startH * width
    val frameLength = cropWidth * cropHeight
    var i = 0
    var c = 0
    val channels = 3
    while (c < channels) {
      i = 0
      while (i < frameLength) {
        val th = i / cropWidth
        val tw = i % cropWidth

        val data_index = (c * height + startH + th) * width + startW + tw
        val top_index = if (mirror) {
          (c * cropHeight + th) * cropWidth + (cropWidth - 1 - tw)
        } else {
          (c * cropHeight + th) * cropWidth + tw
        }
        target(top_index) = source(data_index)
        i += 1
      }
      c += 1
    }
  }
}