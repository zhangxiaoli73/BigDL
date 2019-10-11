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


package com.intel.analytics.bigdl.models.maskrcnn

import breeze.linalg.{*, dim, max}
import com.intel.analytics.bigdl.nn.ResizeBilinear
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.sun.xml.internal.bind.v2.TODO

import scala.collection.mutable.ArrayBuffer

object MaskRCNNUtils {

  // boxes with one dim
  def expandBoxes(boxes: Tensor[Float], scale: Float): Tensor[Float] = {
    val box0 = boxes.valueAt(1)
    val box1 = boxes.valueAt(2)
    val box2 = boxes.valueAt(3)
    val box3 = boxes.valueAt(4)

    var w_half = (box2 - box0) * 0.5f
    var h_half = (box3  - box1) * 0.5f
    val x_c = (box2 + box0) * 0.5f
    val y_c = (box3 + box1) * 0.5f

    w_half *= scale
    h_half *= scale

    val boxes_exp = Tensor[Float]().resizeAs(boxes)
    boxes_exp.setValue(1, x_c - w_half)
    boxes_exp.setValue(3, x_c + w_half)
    boxes_exp.setValue(2, y_c - h_half)
    boxes_exp.setValue(4, y_c + h_half)
    return boxes_exp
  }

  // mask with three dims (channel, height, wide)
  def expandMasks(mask: Tensor[Float], padding: Int): (Tensor[Float], Float) = {
    require(mask.isContiguous(), "Only support contiguous mask")

    val N = mask.size(1)
    val M = mask.size(mask.dim() - 1)
    val pad2 = 2 * padding
    val scale = (M + pad2).toFloat / M
    val paddedMask = Tensor[Float](N, M + pad2, M + pad2)

    val maskHeight = mask.size(2)
    val maskWidth = mask.size(3)
    val padHeight = paddedMask.size(2)
    val padWidth = paddedMask.size(3)

    var i = 1
    while (i <= N) {
      val maskPart = mask.select(1, i)
      val maskArray = maskPart.storage().array()
      val maskOffset = maskPart.storageOffset() - 1

      val padPart = paddedMask.select(1, i)
      val padArray = padPart.storage().array()
      val padOffset = padPart.storageOffset() - 1

      val nElement = padPart.nElement()
      var j = 0
      while (j < nElement) {
        val tempHeight = j / padWidth + 1
        val tempWidth = j % padWidth + 1
        val tempMaskHeight =
          if ((tempHeight > padding + maskHeight) || (tempHeight < padding)) -1
          else tempHeight - padding

        val tempMaskWidth =
          if ((tempWidth > padding + maskWidth) || (tempWidth < padding)) -1
          else tempWidth - padding

        if (tempMaskHeight > 0 && tempMaskWidth > 0) {
          val offset = (tempMaskHeight - 1) * maskWidth + tempMaskWidth - 1
          padArray(j + padOffset) = maskArray(offset + maskOffset)
        }
        j += 1
      }
      i += 1
    }
    (paddedMask, scale)
  }

  // mask and box should be one by one
  def pasteMaskInImage(mask: Tensor[Float], box: Tensor[Float],
    thresh: Float = 0.5f, padding : Int = 1, binaryMask: Tensor[Float]): Unit = {

    val (paddedMask, scale) = expandMasks(mask, padding)
    val boxExpand = expandBoxes(box, scale)

    val TO_REMOVE = 1
    val w = math.max(boxExpand.valueAt(3).toInt - boxExpand.valueAt(1).toInt + TO_REMOVE, 1)
    val h = math.max(boxExpand.valueAt(4).toInt - boxExpand.valueAt(2).toInt + TO_REMOVE, 1)

    paddedMask.resize(1, paddedMask.size(2), paddedMask.size(3))
    val interpMask = Tensor[Float](1, h, w)
    bilinear(paddedMask, interpMask)

    if (thresh >= 0) {
      interpMask.apply1(m => if (m > thresh) 1 else 0)
    } else {
      interpMask.mul(255.0f)
    }

    val imgHeight = binaryMask.size(1)
    val imgWide = binaryMask.size(2)

    val x_0 = math.max(boxExpand.valueAt(1).toInt, 0)
    val x_1 = math.min(boxExpand.valueAt(3).toInt + 1, imgWide)
    val y_0 = math.max(boxExpand.valueAt(2).toInt, 0)
    val y_1 = math.min(boxExpand.valueAt(4).toInt + 1, imgHeight)

    val maskX0 = y_0 - boxExpand.valueAt(2).toInt
    val maskX1 = y_1 - boxExpand.valueAt(2).toInt
    val maskY0 = x_0 - boxExpand.valueAt(1).toInt
    val maskY1 = x_1 - boxExpand.valueAt(1).toInt

    binaryMask.narrow(1, y_0 + 1, y_1 - y_0).narrow(2, x_0 + 1, x_1 - x_0).copy(
      interpMask.narrow(2, maskX0 + 1, maskX1 - maskX0).narrow(3, maskY0 + 1, maskY1 - maskY0))
  }

  // input & output should be 3 dims with (n, height, width)
  def bilinear(input: Tensor[Float], output: Tensor[Float],
               alignCorners: Boolean = false): Unit = {
    val input_height = input.size(2)
    val input_width = input.size(3)
    val output_height = output.size(2)
    val output_width = output.size(3)

    if (input_height == output_height && input_width == output_width) {
     output.copy(input)
     return
    }

    require(input.isContiguous() && output.isContiguous(),
      "Only support contiguous tensor for bilinear")

    val channels = input.size(1)
    val idata = input.storage().array()
    val odata = output.storage().array()
    val ioffset = input.storageOffset() - 1
    val ooffset = output.storageOffset() - 1

    val rheight = areaPixelComputeScale(
      input_height, output_height, alignCorners)

    val rwidth = areaPixelComputeScale(
      input_width, output_width, alignCorners)

    for (h2 <- 0 to (output_height - 1)) {
      val h1r = areaPixelComputeSourceIndex(rheight, h2, alignCorners)

      val h1 = h1r.toInt
      val h1p = if (h1 < input_height - 1) 1 else 0
      val h1lambda = h1r - h1
      val h0lambda = 1.0f - h1lambda

      for (w2 <- 0 to (output_width - 1)) {
        val w1r = areaPixelComputeSourceIndex(
          rwidth, w2, alignCorners)

        val w1 = w1r.toInt
        val w1p = if (w1 < input_width - 1) 1 else 0

        val w1lambda = w1r - w1
        val w0lambda = 1.0f - w1lambda

        val pos1 = h1 * input_width + w1 + ioffset
        val pos2 = h2 * output_width + w2 + ooffset

        //  todo: now only support 1 channel
        for (c <- 0 to (channels - 1)) {
          odata(pos2) = h0lambda * (w0lambda * idata(pos1) + w1lambda * idata(pos1 + w1p)) +
            h1lambda * (w0lambda * idata(pos1 + h1p * input_width) +
              w1lambda * idata(pos1 + h1p * input_width + w1p))
        }
      }
    }
  }

  private def areaPixelComputeScale(
    inputSize: Int, outputSize: Int, alignCorners: Boolean): Float = {
    if (alignCorners) {
      (inputSize - 1).toFloat / (outputSize - 1)
    } else {
      (inputSize.toFloat) / outputSize
    }
  }

  private def areaPixelComputeSourceIndex(
    scale: Float, dstIndex: Int, alignCorners : Boolean) : Float = {
    if (alignCorners) {
      scale * dstIndex
    } else {
      val src_idx = scale * (dstIndex + 0.5f) - 0.5f
      if (src_idx < 0) 0.0f else src_idx
    }
  }
}
