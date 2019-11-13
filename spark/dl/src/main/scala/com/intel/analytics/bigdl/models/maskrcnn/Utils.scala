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

import breeze.linalg.{*, dim, max, min}
import breeze.numerics.{floor, round}
import com.intel.analytics.bigdl.nn.{Bilinear, ResizeBilinear}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.sun.xml.internal.bind.v2.TODO

import scala.collection.mutable.ArrayBuffer

private[bigdl] object Utils {
  // box with 4 element (xyxy)
  def expandBoxes(bbox: Tensor[Float], bboxExpand: Tensor[Float], scale: Float)
  : Unit = {
    require(bbox.nElement() == 4 && bboxExpand.nElement() == 4
      && bbox.dim() == 1 && bboxExpand.dim() == 1,
      "Box and expanded box should have 4 elements with one dim")

    val box0 = bbox.valueAt(1)
    val box1 = bbox.valueAt(2)
    val box2 = bbox.valueAt(3)
    val box3 = bbox.valueAt(4)

    var wHalf = (box2 - box0) * 0.5f
    var hHalf = (box3  - box1) * 0.5f
    val x_c = (box2 + box0) * 0.5f
    val y_c = (box3 + box1) * 0.5f

    wHalf *= scale
    hHalf *= scale

    bboxExpand.setValue(1, x_c - wHalf)
    bboxExpand.setValue(3, x_c + wHalf)
    bboxExpand.setValue(2, y_c - hHalf)
    bboxExpand.setValue(4, y_c + hHalf)
  }

  // mask with three dims (channel, height, wide)
  def expandMasks(mask: Tensor[Float], padding: Int): (Tensor[Float], Float) = {
    require(mask.isContiguous(), "Only support contiguous mask")

    val channel = mask.size(1)
    val width = mask.size(mask.dim() - 1) // height equals to width
    val expandPadding = 2 * padding
    val scale = (width + expandPadding).toFloat / width
    val paddedMask = Tensor[Float](channel, width + expandPadding, width + expandPadding)

    val maskHeight = mask.size(2)
    val maskWidth = mask.size(3)
    val padHeight = paddedMask.size(2)
    val padWidth = paddedMask.size(3)

    for (i <- 1 to  channel) {
      val maskPart = mask.select(1, i)
      val maskArray = maskPart.storage().array()
      val maskOffset = maskPart.storageOffset() - 1

      val padPart = paddedMask.select(1, i)
      val padArray = padPart.storage().array()
      val padOffset = padPart.storageOffset() - 1

      val nElement = padPart.nElement()
      for (j <- 0 until nElement) {
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
      }
    }
    (paddedMask, scale)
  }

  // mask and box should be one by one
  def decodeMaskInImage(mask: Tensor[Float], box: Tensor[Float], binaryMask: Tensor[Float],
    thresh: Float = 0.5f, padding : Int = 1): Unit = {

    val (paddedMask, scale) = expandMasks(mask, padding)
    val boxExpand = Tensor[Float]().resizeAs(box)
    expandBoxes(box, boxExpand, scale)

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

  // mask and box should be one by one
//  def pasteMaskInImage(mask: Tensor[Float], box: Tensor[Float], binaryMask: Tensor[Float],
//                        thresh: Float = 0.5f, padding : Int = 1): Unit = {
//
//    val height = binaryMask.size(1)
//    val width = binaryMask.size(2)
//
//    // do paste mask
//    val x0 = box.valueAt(1)
//    val y0 = box.valueAt(2)
//    val x1 = box.valueAt(3)
//    val y1 = box.valueAt(4)
//
//    val x0_int = math.max(math.floor(x0) - 1, 0).toInt
//    val y0_int = math.max(math.floor(y0) - 1, 0).toInt
//
//    val x1_int = math.min(math.ceil(x1) + 1, height).toInt
//    val y1_int = math.min(math.ceil(y1) + 1, width).toInt
//
//    val img_y = (0 until (y1_int - y0_int)).map(v => {
//      val t = v + 0.5
//      (t - y0) / (y1 - y0) * 2 - 1
//    }).toArray
//    val img_x = (0 until (x1_int - x0_int)).map(v => {
//      val t = v + 0.5
//      (t - x0) / (x1 - x0) * 2 - 1
//    }).toArray
//
//    val graid
//
//    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
//    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
//    grid = torch.stack([gx, gy], dim=3)
//
//
//    val (paddedMask, scale) = expandMasks(mask, padding)
//    val boxExpand = Tensor[Float]().resizeAs(box)
//    expandBoxes(box, boxExpand, scale)
//
//    val w = math.max(boxExpand.valueAt(3).toInt - boxExpand.valueAt(1).toInt + 1, 1)
//    val h = math.max(boxExpand.valueAt(4).toInt - boxExpand.valueAt(2).toInt + 1, 1)
//
//    paddedMask.resize(1, paddedMask.size(2), paddedMask.size(3))
//    val interpMask = Tensor[Float](1, h, w)
//    bilinear(paddedMask, interpMask)
//
//    if (thresh >= 0) {
//      interpMask.apply1(m => if (m > thresh) 1 else 0)
//    } else {
//      interpMask.mul(255.0f)
//    }
//
//    val imgHeight = binaryMask.size(1)
//    val imgWide = binaryMask.size(2)
//
//    val x_0 = math.max(boxExpand.valueAt(1).toInt, 0)
//    val x_1 = math.min(boxExpand.valueAt(3).toInt + 1, imgWide)
//    val y_0 = math.max(boxExpand.valueAt(2).toInt, 0)
//    val y_1 = math.min(boxExpand.valueAt(4).toInt + 1, imgHeight)
//
//    val maskX0 = y_0 - boxExpand.valueAt(2).toInt
//    val maskX1 = y_1 - boxExpand.valueAt(2).toInt
//    val maskY0 = x_0 - boxExpand.valueAt(1).toInt
//    val maskY1 = x_1 - boxExpand.valueAt(1).toInt
//
//    binaryMask.narrow(1, y_0 + 1, y_1 - y_0).narrow(2, x_0 + 1, x_1 - x_0).copy(
//      interpMask.narrow(2, maskX0 + 1, maskX1 - maskX0).narrow(3, maskY0 + 1, maskY1 - maskY0))
//  }
//
//  def grid_sampler_3d_cpu_impl(input: Tensor[Float], grid: Tensor[Float],
//    interpolation_mode: String = "bilinear", padding_mode: String = "zero",
//    align_corners: Boolean = false) {
//    val N = input.size(0)
//    val C = input.size(1)
//    val inp_D = input.size(2)
//    val inp_H = input.size(3)
//    val inp_W = input.size(4)
//    val out_D = grid.size(1)
//    val out_H = grid.size(2)
//    val out_W = grid.size(3)
//    val output = Tensor[Float](N, C, out_D, out_H, out_W)
//
//    val inp_sN = input.stride(0)
//    val inp_sC = input.stride(1)
//    val inp_sD = input.stride(2)
//    val inp_sH = input.stride(3)
//    val inp_sW = input.stride(4)
//    val grid_sN = grid.stride(0)
//    val grid_sD = grid.stride(1)
//    val grid_sH = grid.stride(2)
//    val grid_sW = grid.stride(3)
//    val grid_sCoor = grid.stride(4)
//    val out_sN = output.stride(0)
//    val out_sC = output.stride(1)
//    val out_sD = output.stride(2)
//    val out_sH = output.stride(3)
//    val out_sW = output.stride(4)
//
//    val inp_ptr = input.storage().array()
//    val out_ptr = output.storage().array()
//    val grid_ptr = grid.storage().array()
//
//    for (n <- 0 until N) {
//      val grid_ptr_N = n * grid_sN // grid_ptr(n * grid_sN)
//      val inp_ptr_N = n * inp_sN // inp_ptr(n * inp_sN)
//      for (d <- 0 until out_D) {
//        for (h <- 0 until out_H) {
//          for (w <- 0 until out_W) {
//            // get the corresponding input x, y, z co-ordinates from grid
//            val grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
//            val ix = grid_ptr(grid_ptr_NDHW)
//            val iy = grid_ptr(grid_ptr_NDHW + grid_sCoor)
//            val iz = grid_ptr(grid_ptr_NDHW + 2 * grid_sCoor)
//
//            ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
//            iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
//            iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);
//
//            if (interpolation_mode == "bilinear") {
//              // get corner pixel values from (x, y, z)
//              // for 4d, we used north-east-south-west
//              // for 5d, we add top-bottom
//              val ix_tnw = math.floor(ix).toInt
//              val iy_tnw = math.floor(iy).toInt
//              val iz_tnw = math.floor(iz).toInt
//
//              val ix_tne: Int = ix_tnw + 1
//              val iy_tne: Int = iy_tnw
//              val iz_tne: Int = iz_tnw
//
//              val ix_tsw: Int = ix_tnw
//              val iy_tsw: Int = iy_tnw + 1
//              val iz_tsw: Int = iz_tnw
//
//              val ix_tse: Int = ix_tnw + 1
//              val iy_tse: Int = iy_tnw + 1
//              val iz_tse: Int = iz_tnw
//
//              val ix_bnw: Int = ix_tnw
//              val iy_bnw: Int = iy_tnw
//              val iz_bnw: Int = iz_tnw + 1
//
//              val ix_bne: Int = ix_tnw + 1
//              val iy_bne: Int = iy_tnw
//              val iz_bne: Int = iz_tnw + 1
//
//              val ix_bsw: Int = ix_tnw
//              val iy_bsw: Int = iy_tnw + 1
//              val iz_bsw: Int = iz_tnw + 1
//
//              val ix_bse: Int = ix_tnw + 1
//              val iy_bse: Int = iy_tnw + 1
//              val iz_bse: Int = iz_tnw + 1
//
//              // get surfaces to each neighbor:
//              val tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
//              val tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
//              val tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
//              val tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
//              val bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
//              val bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
//              val bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
//              val bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);
//
//              // calculate bilinear weighted pixel value and set output pixel
//              var out_ptr_NCDHW = n * out_sN + d * out_sD + h * out_sH + w * out_sW // out_ptr
//              var inp_ptr_NC = inp_ptr_N
//              for (c <- 0 until C) {
//                out_ptr_NCDHW += out_sC
//                inp_ptr_NC += inp_sC
//
//                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW) * tnw
//                }
//                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW) * tne
//                }
//                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW) * tsw
//                }
//                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW) * tse
//                }
//                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW) * bnw
//                }
//                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW) * bne
//                }
//                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW) * bsw
//                }
//                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
//                  out_ptr(out_ptr_NCDHW) += inp_ptr(inp_ptr_NC + iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW) * bse
//                }
//              }
//            }
//          }
//        }
//      }
//    return output
//  }


  // input & output should be 3 dims with (n, height, width)
  def bilinear(input: Tensor[Float], output: Tensor[Float],
               alignCorners: Boolean = false): Unit = {
    require(input.dim() == 3 && output.dim() == 3, s"Only support 3 dims bilinear," +
      s"but get ${input.dim()} ${output.dim()}")

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
    val inputData = input.storage().array()
    val outputData = output.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputOffset = output.storageOffset() - 1

    val realHeight = areaPixelComputeScale(
      input_height, output_height, alignCorners)
    val realWidth = areaPixelComputeScale(
      input_width, output_width, alignCorners)

    for (h2 <- 0 until output_height) {
      val h1r = areaPixelComputeSourceIndex(realHeight, h2, alignCorners)
      val h1 = h1r.toInt
      val h1p = if (h1 < input_height - 1) 1 else 0
      val h1lambda = h1r - h1
      val h0lambda = 1.0f - h1lambda

      for (w2 <- 0 until output_width) {
        val w1r = areaPixelComputeSourceIndex(realWidth, w2, alignCorners)
        val w1 = w1r.toInt
        val w1p = if (w1 < input_width - 1) 1 else 0
        val w1lambda = w1r - w1
        val w0lambda = 1.0f - w1lambda

        val pos1 = h1 * input_width + w1 + inputOffset
        val pos2 = h2 * output_width + w2 + outputOffset

        for (c <- 0 to (channels - 1)) {
          outputData(pos2) = h0lambda * (w0lambda * inputData(pos1) +
            w1lambda * inputData(pos1 + w1p)) +
            h1lambda * (w0lambda * inputData(pos1 + h1p * input_width) +
              w1lambda * inputData(pos1 + h1p * input_width + w1p))
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
      val srcIdx = scale * (dstIndex + 0.5f) - 0.5f
      if (srcIdx < 0) 0.0f else srcIdx
    }
  }
}
