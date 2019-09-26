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

object Mask {

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

  // mask two dims
  def expandMasks(mask: Tensor[Float], padding: Int): (Tensor[Float], Float) = {
    val N = mask.size(1)
    val M = mask.size(mask.dim() - 1)
    val pad2 = 2 * padding
    val scale = (M + pad2).toFloat / M
    val padded_mask = Tensor[Float](N, 1, M + pad2, M + pad2)

    require(mask.isContiguous() && padded_mask.isContiguous())

    val maskHeight = mask.size(2)
    val maskWidth = mask.size(3)
    val padHeight = padded_mask.size(3)
    val padWidth = padded_mask.size(4)

    var i = 1
    while (i <= N) {
      val maskPart = mask.select(1, i)
      val maskArr = maskPart.storage().array()
      val maskOffset = maskPart.storageOffset() - 1

      val padPart = padded_mask.select(1, i)
      val padArr = padPart.storage().array()
      val padOffset = padPart.storageOffset() - 1

      val nElement = padPart.nElement()
      var j = 0
      while (j < nElement) {
        val tempHeight = j / padWidth + 1
        val tempWidth = j % padWidth + 1
        val tempMaskHeight = if ((tempHeight > padding + maskHeight) || (tempHeight < padding)) {
          -1
        } else tempHeight - padding

        val tempMaskWidth = if ((tempWidth > padding + maskWidth) || (tempWidth < padding)) {
          -1
        } else tempWidth - padding

        if (tempMaskHeight > 0 && tempMaskWidth > 0) {
          val m = (tempMaskHeight - 1) * maskWidth + tempMaskWidth - 1
          padArr(j + padOffset) = maskArr(m + maskOffset)
        }
        j += 1
      }
      i += 1
    }
    (padded_mask, scale)
  }

  // mask and box should be one by one
  def pasteMaskInImage(mask: Tensor[Float], box: Tensor[Float],
    im_h: Int, im_w: Int, thresh: Float = 0.5f, padding : Int = 1): Tensor[Float] = {

    val (padded_mask, scale) = expandMasks(mask, padding = padding)
    val boxExpand = expandBoxes(box, scale)

    val TO_REMOVE = 1
    val w = math.max(boxExpand.valueAt(3).toInt - boxExpand.valueAt(1).toInt + TO_REMOVE, 1)
    val h = math.max(boxExpand.valueAt(4).toInt - boxExpand.valueAt(2).toInt + TO_REMOVE, 1)

    padded_mask.resize(1, padded_mask.size(2), padded_mask.size(3))
    val lastMask = Tensor[Float](1, h, w)
    bilinear(padded_mask, lastMask)

    if (thresh >= 0) {
      lastMask.apply1(m => if (m > thresh) 1 else 0)
    } else {
      lastMask.mul(255.0f)
    }

    val im_mask = Tensor[Float](im_h, im_w)
    val x_0 = math.max(boxExpand.valueAt(1).toInt, 0)
    val x_1 = math.min(boxExpand.valueAt(3).toInt + 1, im_w)
    val y_0 = math.max(boxExpand.valueAt(2).toInt, 0)
    val y_1 = math.min(boxExpand.valueAt(4).toInt + 1, im_h)

    val maskX0 = y_0 - boxExpand.valueAt(2).toInt
    val maskX1 = y_1 - boxExpand.valueAt(2).toInt
    val maskY0 = x_0 - boxExpand.valueAt(1).toInt
    val maskY1 = x_1 - boxExpand.valueAt(1).toInt

    val tmp1 = lastMask.narrow(2, maskX0 + 1, maskX1 - maskX0).narrow(3, maskY0 + 1, maskY1 - maskY0)
    val tmp2 = im_mask.narrow(1, y_0 + 1, y_1 - y_0).narrow(2, x_0 + 1, x_1 - x_0)
    tmp2.copy(tmp1)
    return im_mask
  }

  // compute rle iou
  def rleIOU(detection: Array[Tensor[Float]], groud: Array[Tensor[Float]],
             height: Array[Int], width: Array[Int],
             dtNumber: Int, gtNumber: Int,
             iscrowd: Array[Boolean]): Array[Float] = {
    val gtBbox = Tensor[Float](gtNumber, 4)
    val dtBbox = Tensor[Float](dtNumber, 4)

    for (i <- 1 to gtNumber) {
      val box = gtBbox.select(1, i)
      rleToOneBbox(groud(i - 1), box, height(i - 1), width(i -1))
    }

    for (i <- 1 to dtNumber) {
      val box = dtBbox.select(1, i)
      rleToOneBbox(detection(i - 1), box, height(i - 1), width(i - 1))
    }

    val iou = bboxIOU(gtBbox, dtBbox, gtNumber, dtNumber)

    for (g <- 0 to (gtNumber - 1)) {
      for (d <- 0 to (dtNumber - 1)) {
        val n = g * dtNumber + d
        if (iou(n) > 0) {
          val crowd = iscrowd(g)

          val dCnts = detection(d)
          val gCnts = groud(g)

          var a = 1
          var b = 1

          var ca = dCnts.valueAt(a)
          val ka = dCnts.nElement()
          var va : Boolean = false
          var vb : Boolean = false

          var cb = gCnts.valueAt(b)
          val kb = gCnts.nElement()
          var i = 0.0f
          var u = 0.0f
          var ct = 1.0f

          while( ct > 0 ) {
            val c = math.min(ca, cb)
            if(va || vb) {
              u = u + c
              if (va && vb) i += c
            }
            ct = 0

            ca = ca - c
            if(ca <= 0 && a < ka) {
              a += 1
              ca = dCnts.valueAt(a)
              va = !va
            }
            ct += ca

            cb = cb - c
            if( cb <= 0 && b < kb) {
              b += 1
              cb = gCnts.valueAt(b)
              vb = !vb
            }
            ct += cb
          }
          iou(g * gtNumber + d) = i / u
        }
      }
    }
    iou
  }

  //  compute bbox iou, which not same with bigdl
  def bboxIOU(gt: Tensor[Float], dt: Tensor[Float],
              gtNumber: Int, dtNumber: Int): Array[Float] = {
    val res = new ArrayBuffer[Float]

    for( g <- 1 to gtNumber) {
      val gt1 = gt.select(1, g)
      val ga = gt1.valueAt(3) * gt1.valueAt(4)

      for( d <- 1 to dtNumber) {
        val dt1 = dt.select(1, d)
        val da = dt1.valueAt(3) * dt1.valueAt(4)

        val w = math.min(dt1.valueAt(3) + dt1.valueAt(1), gt1.valueAt(3) + gt1.valueAt(1)) - math.max(dt1.valueAt(1), gt1.valueAt(1))

        val h = math.min(dt1.valueAt(4) + dt1.valueAt(2), gt1.valueAt(4) + gt1.valueAt(2)) - math.max(dt1.valueAt(2), gt1.valueAt(2))

        var ii = w * h
        val o = ii / (da + ga - ii)
        res.append(o)
      }
    }
    res.toArray
  }

  // convert one rle to one bbox
  def rleToOneBbox(rle: Tensor[Float], bbox: Tensor[Float],
                height: Int, width: Int): Unit = {
    val m = rle.nElement() / 2 * 2

    val h = height.toFloat
    var xp = 0.0f
    var cc = 0
    var xs = width.toFloat
    var ys = height.toFloat
    var ye = 0.0f
    var xe = 0.0f

    if(m == 0) {
      bbox.fill(0.0f)
    } else {
      for (j <- 0 to (m - 1)) {
        cc += rle.valueAt(j + 1).toInt
        val t = cc - j % 2
        val y = t % h
        val x = (t - y) / h
        if(j%2 == 0) {
          xp = x
        } else if (xp < x) {
          ys = 0
          ye = h -1
        }
        xs = math.min(xs, x)
        xe = math.max(xe, x)
        ys = math.min(ys, y)
        ye = math.max(ye, y)
      }

      bbox.setValue(1, xs)
      bbox.setValue(3, xe- xs)
      bbox.setValue(2, ys)
      bbox.setValue(4, ye - ys + 1)
    }
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

    require(input.isContiguous() && output.isContiguous())
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
      val h1r = areaPixelComputeSourceIndex(
        rheight, h2, alignCorners)

      val h1 : Int = h1r.toInt
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

  def areaPixelComputeScale(
    inputSize: Int, outputSize: Int, alignCorners: Boolean): Float = {
    if (alignCorners) {
      (inputSize - 1).toFloat / (outputSize - 1)
    } else {
      (inputSize.toFloat) / outputSize
    }
  }

  def areaPixelComputeSourceIndex(
    scale: Float, dstIndex: Int, alignCorners : Boolean) : Float = {
    if (alignCorners) {
      scale * dstIndex
    } else {
      val src_idx = scale * (dstIndex + 0.5f) - 0.5f
      if (src_idx < 0) 0.0f else src_idx
    }
  }

  def toImageList(tensors: Array[Tensor[Float]], size_divisible: Int = 0): Tensor[Float] = {
    val channel = tensors(0).size(1)
    var maxHeight = 0
    var maxWide = 0
    for (i <- 0 to (tensors.length - 1)) {
      require(channel == tensors(i).size(1),
        s"tensors in one batch should have same channel, but get ${channel} ${tensors(i).size(1)}")
      if (maxHeight < tensors(i).size(2)) maxHeight = tensors(i).size(2)
      if (maxWide < tensors(i).size(3)) maxWide = tensors(i).size(3)
    }

    if (size_divisible > 0) {
      maxHeight = (math.ceil(maxHeight.toFloat / size_divisible) * size_divisible).toInt
      maxWide = (math.ceil(maxWide.toFloat / size_divisible) * size_divisible).toInt
    }
    val batchSize = tensors.length
    val batchShape = Array(batchSize, channel, maxHeight, maxWide)
    val batchImgs = Tensor[Float](batchShape)

    for (i <- 0 to (tensors.length - 1)) {
      batchImgs.select(1, 1).narrow(2, 1, tensors(i).size(2))
        .narrow(3, 1, tensors(i).size(3)).copy(tensors(i))
    }
    batchImgs
  }
}
