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
import com.intel.analytics.bigdl.models.maskrcnn.{MaskRCNN, MaskUtils}
import com.intel.analytics.bigdl.nn.ResizeBilinear
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelNormalize, ColorJitter, Resize}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
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

  def prepareForCocoDetection() : Unit = {
    // need, category_id, bbox, score
  }

  // box shape: box_number * 4
  // mask shape: box_number * 1* 28 * 28
  def prepareForCocoSegmentation(mask: Tensor[Float], bbox: Tensor[Float],
                                 imageHeight: Int, imageWidth: Int): Unit = {
    // need, category_id, segmentation, score

    // resize mask
    require(mask.size(1) == bbox.size(1), s"error get ${mask.size(1)} ${bbox.size(1)}")
    val boxNumber = mask.size(1)
    var i = 0
    while (i < boxNumber) {

      i += 1
    }


    // encode to rle
  }

  def expandBoxes(boxes: Tensor[Float], scale: Float): Tensor[Float] = {
    val box0 = boxes.select(2, 1)
    val box1 = boxes.select(2, 2)
    val box2 = boxes.select(2, 3)
    val box3 = boxes.select(2, 4)

    var w_half = Tensor[Float]().resizeAs(box2).copy(box2).sub(box0).mul(0.5f)
    var h_half = Tensor[Float]().resizeAs(box3).copy(box3).sub(box1).mul(0.5f)
    val x_c = Tensor[Float]().resizeAs(box2).copy(box2).add(box0).mul(0.5f)
    val y_c = Tensor[Float]().resizeAs(box3).copy(box3).add(box1).mul(0.5f)

    w_half *= scale
    h_half *= scale

    val boxes_exp = Tensor[Float]().resizeAs(boxes)
    boxes_exp.select(2, 1).copy(x_c - w_half)
    boxes_exp.select(2, 3).copy(x_c + w_half)
    boxes_exp.select(2, 2).copy(y_c - h_half)
    boxes_exp.select(2, 4).copy(y_c + h_half)
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

  def pasteMaskInImage(mask: Tensor[Float], box: Tensor[Float],
    im_h: Int, im_w: Int, thresh: Float = 0.5f, padding : Int = 1): Tensor[Float] = {

    val (padded_mask, scale) = expandMasks(mask, padding = padding)
    val boxExpand = expandBoxes(box, scale)

    val TO_REMOVE = 1
    val w = math.max(boxExpand.valueAt(1, 3).toInt - boxExpand.valueAt(1, 1).toInt + TO_REMOVE, 1)
    val h = math.max(boxExpand.valueAt(1, 4).toInt - boxExpand.valueAt(1, 2).toInt + TO_REMOVE, 1)

    padded_mask.resize(1, 1, padded_mask.size(2), padded_mask.size(3))
    val lastMask = bilinearResize(padded_mask, Array(h, w))
    // bilinearResize(mask, Array(40, 50))

    if (thresh >= 0) {
      lastMask.apply1(m => if (m > thresh) 1 else 0)
    } else {
      lastMask.mul(255.0f)
    }

    val im_mask = Tensor[Float](im_h, im_w)
    val x_0 = math.max(boxExpand.valueAt(1, 1).toInt, 0)
    val x_1 = math.min(boxExpand.valueAt(1, 3).toInt + 1, im_w)
    val y_0 = math.max(boxExpand.valueAt(1, 2).toInt, 0)
    val y_1 = math.min(boxExpand.valueAt(1, 4).toInt + 1, im_h)

    val maskX0 = y_0 - boxExpand.valueAt(1, 2).toInt
    val maskX1 = y_1 - boxExpand.valueAt(1, 2).toInt
    val maskY0 = x_0 - boxExpand.valueAt(1, 1).toInt
    val maskY1 = x_1 - boxExpand.valueAt(1, 1).toInt

    val tmp1 = lastMask.narrow(3, maskX0 + 1, maskX1 - maskX0).narrow(4, maskY0 + 1, maskY1 - maskY0)
    val tmp2 = im_mask.narrow(1, y_0 + 1, y_1 - y_0).narrow(2, x_0 + 1, x_1 - x_0)
    tmp2.copy(tmp1)
//    im_mask[y_0:y_1, x_0:x_1] = mask[
//      (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
//    ]
    return im_mask
  }

  def bilinearResize(input: Tensor[Float], size: Array[Int],
                     align_corners: Boolean = false): Tensor[Float] = {
    val module = ResizeBilinear[Float](
      size(0),
      size(1),
      false,
      dataFormat = DataFormat.NCHW
    )

    module.forward(input)
  }

  def rleEncode(binaryMask: Tensor[Float]): Tensor[Int] = {
    val countsBuffer = new ArrayBuffer[Int]

    val h = binaryMask.size(1)
    val w = binaryMask.size(2)
    val maskArr = binaryMask.storage().array()
    val offset = binaryMask.storageOffset() - 1

    val n = binaryMask.nElement()
    var i = 0
    var p = -1
    var c = 0
    while (i < n) {
      // the first one should be 0
      if (p == -1 && maskArr(i + offset) == 1) {
        countsBuffer.append(0)
        p = 1
        c = 1
      } else if (p == -1 && maskArr(i + offset) == 0) {
        p = 0
        c = 1
      } else if (maskArr(i + offset) == p) {
        c += 1
      } else {
        countsBuffer.append(c)
        c = 1
        p = maskArr(i + offset).toInt
      }
      i += 1
    }
    countsBuffer.append(c)

    val out = Tensor[Int](countsBuffer.length)

    System.arraycopy(countsBuffer.toArray, 0, out.storage().array(), 0, countsBuffer.length)

    out
  }

  // array length is bbox number
  def rleIOU(detection: Array[Tensor[Float]], groud: Array[Tensor[Float]],
                   height: Array[Int], width: Array[Int],
                   dtNumber: Int, gtNumber: Int,
                   iscrowd: Array[Boolean]): Array[Float] = {
    val gtBbox = Tensor[Float](gtNumber, 4)
    val dtBbox = Tensor[Float](dtNumber, 4)

    for (i <- 1 to gtNumber) {
      val box = gtBbox.select(1, i)
      rleToBbox(groud(i - 1), box, height(i - 1), width(i -1))
    }

    for (i <- 1 to dtNumber) {
      val box = dtBbox.select(1, i)
      rleToBbox(detection(i - 1), box, height(i - 1), width(i - 1))
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
          var va = 0
          var vb = 0

          var cb = gCnts.valueAt(b)
          val kb = gCnts.nElement()
          var i = 0.0f
          var u = 0.0f
          var ct = 1.0f

          while( ct > 0 ) {
            val c = math.min(ca, cb)
            if(va > 0 || vb > 0) {
              u = u + c
              if (va > 0 && vb > 0) {
                i += c
              }
            } else ct = 0

            ca = ca - c
            if(ca <= 0 && a <= ka) {
//              a += 1
//              ca = dCnts.valueAt(a)

//              ca = dCnts.valueAt(a)
//              a += 1
              va = 1 - va
            } else ct += ca

            cb = cb - c
            if( cb <= 0 && b <= kb) {
//              b += 1
//              cb = gCnts.valueAt(b)

              cb = gCnts.valueAt(b)
              b += 1

              vb = 1- vb
            } else ct += cb
          }
          iou(g * gtNumber + d) = i.toFloat / u
        }
      }
    }

    iou
  }


//  def rleArea( rle: Tensor[Float], n: Int, uint *a ) {
//    siz i, j; for( i=0; i<n; i++ ) {
//      a[i]=0; for( j=1; j<R[i].m; j+=2 ) a[i]+=R[i].cnts[j]; }
//  }

//  def bboxIOU(gt: Tensor[Float], dt: Tensor[Float],
//              gtNumber: Int, dtNumber: Int): Array[Float] = {
//    val res = new ArrayBuffer[Float]
//
//    for( g <- 1 to gtNumber) {
//      val gt1 = gt.select(1, g)
//      for( d <- 1 to dtNumber) {
//        val dt1 = dt.select(1, d)
//        res.append(getIOURate(gt1, dt1))
//      }
//    }
//    res.toArray
//  }

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

//  def getIOURate(bbox1: Tensor[Float],
//                 bbox2: Tensor[Float]): Float = {
//    val x1 = bbox2.valueAt(1)
//    val y1 = bbox2.valueAt(2)
//    val x2 = bbox2.valueAt(3)
//    val y2 = bbox2.valueAt(4)
//
//    val xmin = bbox1.valueAt(1)
//    val ymin = bbox1.valueAt(2)
//    val xmax = bbox1.valueAt(3)
//    val ymax = bbox1.valueAt(4)
//
//    val area = (xmax - xmin) * (ymax - ymin)
//
//    val ixmin = Math.max(xmin, x1)
//    val iymin = Math.max(ymin, y1)
//    val ixmax = Math.min(xmax, x2)
//    val iymax = Math.min(ymax, y2)
//    val inter = Math.max(ixmax - ixmin, 0) * Math.max(iymax - iymin, 0)
//
//    // special process for crowd
//    inter / ((x2 - x1) * (y2 - y1) + area - inter)
//  }

  def getIOURate(bbox1: Tensor[Float],
                 bbox2: Tensor[Float]): Float = {
    val x1 = bbox2.valueAt(1)
    val y1 = bbox2.valueAt(2)
    val x2 = bbox2.valueAt(3)
    val y2 = bbox2.valueAt(4)

    val xmin = bbox1.valueAt(1)
    val ymin = bbox1.valueAt(2)
    val xmax = bbox1.valueAt(3)
    val ymax = bbox1.valueAt(4)

    val area = (xmax - xmin) * (ymax - ymin)

    val ixmin = Math.max(xmin, x1)
    val iymin = Math.max(ymin, y1)
    val ixmax = Math.min(xmax, x2)
    val iymax = Math.min(ymax, y2)
    val inter = Math.max(ixmax - ixmin, 0) * Math.max(iymax - iymin, 0)

    // special process for crowd
    inter / ((x2 - x1) * (y2 - y1) + area - inter)
  }

  def rleToBbox(rle: Tensor[Float], bbox: Tensor[Float],
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
        var t = cc - j % 2
        var y = t % h
        var x = (t - y).toFloat / h
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
}


