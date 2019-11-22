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
import com.intel.analytics.bigdl.utils.{T, Table}
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
  def decodeMaskInImageOld(mask: Tensor[Float], box: Tensor[Float], binaryMask: Tensor[Float],
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

    println(interpMask.narrow(2, maskX0 + 1, maskX1 - maskX0).narrow(3, maskY0 + 1, maskY1 - maskY0))
  }

  // mask and box should be one by one
  def decodeMaskInImage(mask: Tensor[Float], box: Tensor[Float],
    binaryMask: Tensor[Float], thresh: Float = 0.5f): Unit = {

    val height = binaryMask.size(1)
    val width = binaryMask.size(2)

    // do paste mask
    val x0 = box.valueAt(1)
    val y0 = box.valueAt(2)
    val x1 = box.valueAt(3)
    val y1 = box.valueAt(4)

    val x0_int = math.max(math.floor(x0) - 1, 0).toInt
    val y0_int = math.max(math.floor(y0) - 1, 0).toInt

    val x1_int = math.min(math.ceil(x1) + 1, width).toInt
    val y1_int = math.min(math.ceil(y1) + 1, height).toInt

    val img_y = (y0_int until y1_int).map(v => {
      val t = v + 0.5
      val t2 = (t - y0) / (y1 - y0) * 2 - 1
      t2
    }).toArray
    val img_x = (x0_int until x1_int).map(v => {
      val t = v + 0.5
      (t - x0) / (x1 - x0) * 2 - 1
    }).toArray

    val N = mask.size(1)
    // val grid = Tensor[Float](N, img_y.length, img_x.length, 2)
    val grid = Tensor[Float](img_y.length, img_x.length, 2)

    val gridArray = grid.storage().array()
    val gridOffset = grid.storageOffset() - 1
    var yOffset = 0
    var i = 0
    while (i < grid.nElement()) {
      for (j <- 0 until img_x.length) {
        gridArray(gridOffset + i) = img_x(j).toFloat
        gridArray(gridOffset + i + 1) = img_y(yOffset).toFloat
        i += 2
      }
      yOffset += 1
    }

    val output = Tensor[Float]()
    val output2 = Tensor[Float]()

    gridSamplerWithBilinear(mask, grid, output)
    gridSamplerWithBilinearWithCorners(mask, grid, output2, alignCorners = true)

    if (thresh >= 0) {
      output.apply1(m => if (m > thresh) 1 else 0)
    } else {
      output.mul(255.0f)
    }

    binaryMask.narrow(1, y0_int + 1, y1_int - y0_int)
      .narrow(2, x0_int + 1, x1_int - x0_int).copy(output)
  }

  def computeInterpParams(x: Float, y: Float, inp_W: Int, inp_H: Int): Table = {
    // get NE, NW, SE, SW pixel values from (x, y)
    // assuming we get exact integer representation and just use scalar_t
    // if we don't, the weights will be garbage anyways.
    val x_w: Int = x.toInt
    val y_n: Int = y.toInt

    // get distances to each side
    val w: Float = x - x_w
    val e: Float = 1.0f - w
    val n: Float = y - y_n
    val s: Float = 1.0f - n

    // get interpolation weights for each neighbor
    // e.g., for the nw corder, the weight is `dist_to_south * dist_to_east`.
    val nw = s * e
    val ne = s * w
    val sw = n * e
    val se = n * w

    val i_x_w = x_w.toInt
    val i_y_n = y_n.toInt
    val i_x_e = i_x_w + 1
    val i_y_s = i_y_n + 1

    // Use int comparison because it is much faster than float comp with AVX2
    // (latency 1 cyc vs. 4 cyc on skylake)
    // Avoid using the le and ge because those are not implemented in AVX2 and
    // are actually simulated using multiple instructions.
    //  must_in_bound = padding != GridSamplerPadding::Zeros, so here it is false
    val w_mask = if ((i_x_w > -1) & (i_x_w < inp_W)) 1 else 0
    val n_mask = if ((i_y_n > -1) & (i_y_n < inp_H)) 1 else 0
    val e_mask = if ((i_x_e > -1) & (i_x_e < inp_W)) 1 else 0
    val s_mask = if ((i_y_s > -1) & (i_y_s < inp_H)) 1 else 0
    val nw_mask = w_mask & n_mask
    val ne_mask = e_mask & n_mask
    val sw_mask = w_mask & s_mask
    val se_mask = e_mask & s_mask

    T(n, s, w, e, // distances to 4 sides
      nw, ne, sw, se,  // interpolation weights wrt 4 corners
      nw_mask, ne_mask, sw_mask, se_mask,  // in_bound masks
      i_y_n, i_x_w) // y_n and x_w
  }

  def computeLocation(size: Int, in: Float, alignCorners: Boolean = false): Float = {
    val max_val = size - 1
    val scaling_factor = if (alignCorners) (size - 1).toFloat / 2 else size.toFloat / 2
    val low = if (alignCorners) 0f else -0.5f
    val twice_span = if (alignCorners) (size - 1) * 2 else size * 2
    val empty = size <= 0

    if (alignCorners) {
      (in + 1) * scaling_factor
    } else (in + 1) * scaling_factor - 0.5f
  }

  def gridSamplerWithBilinearWithCorners(input: Tensor[Float], grid: Tensor[Float],
    output: Tensor[Float], alignCorners: Boolean = false): Unit = {

    val C = input.size(1)
    val IH = input.size(2)
    val IW = input.size(3)
    val H = grid.size(1)
    val W = grid.size(2)

    output.resize(C, H, W)

    for (h <- 0 until H) {
      for (w <- 0 until W) {
        // get the corresponding input x, y co-ordinates from grid
        val x = grid.valueAt(h + 1, w + 1, 1)
        val y = grid.valueAt(h + 1, w + 1, 2)

        val compute_x = computeLocation(IW, x, alignCorners)
        val compute_y = computeLocation(IH, y, alignCorners)

        val interp_params = computeInterpParams(compute_x, compute_y, IW, IH)

        val nw = interp_params.get[Float](5).get
        val ne = interp_params.get[Float](6).get
        val sw = interp_params.get[Float](7).get
        val se = interp_params.get[Float](8).get

        val nw_mask = interp_params.get[Int](9).get
        val ne_mask = interp_params.get[Int](10).get
        val sw_mask = interp_params.get[Int](11).get
        val se_mask = interp_params.get[Int](12).get

        val i_y_n = interp_params.get[Int](13).get
        val i_x_w = interp_params.get[Int](14).get

        val i_nw_offset = i_y_n * IH + i_x_w * IW
        val i_ne_offset = i_nw_offset + IW
        val i_sw_offset = i_nw_offset + IH
        val i_se_offset = i_sw_offset + IW


        for (c <- 0 until C) {
          val nw_val = safeGet(input, c, i_nw_offset.toInt, nw_mask) // i_nw, nw_mask
          val ne_val = safeGet(input, c, i_ne_offset.toInt, ne_mask)
          val sw_val = safeGet(input, c, i_ne_offset.toInt, sw_mask)
          val se_val = safeGet(input, c, i_se_offset.toInt, se_mask)
          val out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se
          output.setValue(c + 1, h + 1, w + 1, out_val)
        }
      }
    }
  }

  def gridSamplerWithBilinear(input: Tensor[Float], grid: Tensor[Float],
                              output: Tensor[Float]): Tensor[Float] = {
    val C = input.size(1)
    val IH = input.size(2)
    val IW = input.size(3)
    val H = grid.size(1)
    val W = grid.size(2)

    output.resize(C, H, W)
    for (h <- 0 until H) {
      for (w <- 0 until W) {
        // get the corresponding input x, y co-ordinates from grid
        var ix = grid.valueAt(h + 1, w + 1, 1)
        var iy = grid.valueAt(h + 1, w + 1, 2)

        // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
        ix = ((ix + 1) / 2) * (IW-1)
        iy = ((iy + 1) / 2) * (IH-1)

        // for align corners
//        ix = ((ix + 1) / 2) * IW - 0.5f
//        iy = ((iy + 1) / 2) * IH - 0.5f

        // get NE, NW, SE, SW pixel values from (x, y)
        val ix_nw: Int = ix.toInt
        val iy_nw: Int = iy.toInt
        val ix_ne: Int = ix_nw + 1
        val iy_ne: Int = iy_nw
        val ix_sw: Int = ix_nw
        val iy_sw: Int = iy_nw + 1
        val ix_se: Int = ix_nw + 1
        val iy_se: Int = iy_nw + 1

        // get surfaces to each neighbor:
        val nw = (ix_se - ix) * (iy_se - iy)
        val ne = (ix    - ix_sw) * (iy_sw - iy)
        val sw = (ix_ne - ix) * (iy    - iy_ne)
        val se = (ix    - ix_nw) * (iy    - iy_nw)

        // calculate bilinear weighted pixel value and set output pixel
        for (c <- 0 until C) {
          val nw_val = safeGet(input, c, iy_nw, ix_nw)
          val ne_val = safeGet(input, c, iy_ne, ix_ne)
          val sw_val = safeGet(input, c, iy_sw, ix_sw)
          val se_val = safeGet(input, c, iy_se, ix_se)
          val out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se
          output.setValue(c + 1, h + 1, w + 1, out_val)
        }
      }
    }

    output
  }
//
//  // input shape (C, H', W'), grid shape (H, W, 2), output shape (C, H, W)
//  def gridSamplerWithBilinear(input: Tensor[Float], grid: Tensor[Float],
//                              output: Tensor[Float]): Tensor[Float] = {
//    val N = input.size(1)
//    val C = input.size(2)
//    val IH = input.size(3)
//    val IW = input.size(4)
//    val H = grid.size(2)
//    val W = grid.size(3)
//
//    output.resize(N, C, H, W)
//    for (n <- 0 until N) {
//      for (h <- 0 until H) {
//        for (w <- 0 until W) {
//          // get the corresponding input x, y co-ordinates from grid
//          var ix = grid.valueAt(n + 1, h + 1, w + 1, 1)
//          var iy = grid.valueAt(n + 1, h + 1, w + 1, 2)
//
//          // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
//          ix = ((ix + 1) / 2) * (IW-1)
//          iy = ((iy + 1) / 2) * (IH-1)
//
//          // get NE, NW, SE, SW pixel values from (x, y)
//          val ix_nw: Int = ix.toInt
//          val iy_nw: Int = iy.toInt
//          val ix_ne: Int = ix_nw + 1
//          val iy_ne: Int = iy_nw
//          val ix_sw: Int = ix_nw
//          val iy_sw: Int = iy_nw + 1
//          val ix_se: Int = ix_nw + 1
//          val iy_se: Int = iy_nw + 1
//
//          // get surfaces to each neighbor:
//          val nw = (ix_se - ix) * (iy_se - iy)
//          val ne = (ix    - ix_sw) * (iy_sw - iy)
//          val sw = (ix_ne - ix) * (iy    - iy_ne)
//          val se = (ix    - ix_nw) * (iy    - iy_nw)
//
//          // calculate bilinear weighted pixel value and set output pixel
//          for (c <- 0 until C) {
//            val nw_val = safeGet(input, n, c, iy_nw, ix_nw)
//            val ne_val = safeGet(input, n, c, iy_ne, ix_ne)
//            val sw_val = safeGet(input, n, c, iy_sw, ix_sw)
//            val se_val = safeGet(input, n, c, iy_se, ix_se)
//            val out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se
//            output.setValue(n + 1, c + 1, h + 1, w + 1, out_val)
//          }
//        }
//      }
//    }
//
//    output
//  }

//  def safeGet(input: Tensor[Float], c: Int, h: Int, w: Int) : Float = {
//    if (h < input.size(3) && h >= 0 && w < input.size(4) && w >= 0) {
//      input.valueAt(c + 1, h + 1, w + 1)
//    } else 0
//  }

  def safeGet(input: Tensor[Float], c: Int, h: Int, w: Int) : Float = {
    if (h < input.size(2) && h >= 0 && w < input.size(3) && w >= 0) {
      input.valueAt(c + 1, h + 1, w + 1)
    } else 0
  }

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
