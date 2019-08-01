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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect._

class RoiAlign[T: ClassTag] (
  val spatio_scale: T,
  val sampling_ratio: Int,
  val pooled_height: Int,
  val pooled_width: Int
) (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T]{
  override def updateOutput(input: Table): Tensor[T] = {
    if (classTag[T] == classTag[Double]) {
      val data = input[Tensor[Double]](1)
      val rois = input[Tensor[Double]](2)
      poolOneRoiDouble(data, rois, spatio_scale.asInstanceOf[Double])
    } else if (classTag[T] == classTag[Float]) {
      val data = input[Tensor[Float]](1)
      val rois = input[Tensor[Float]](2)
      poolOneRoiFloat(data, rois, spatio_scale.asInstanceOf[Float])
    } else {
      throw new IllegalArgumentException("currently only Double and Float types are supported")
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = null
    gradInput
  }

  private def poolOneRoiFloat(data: Tensor[Float], rois: Tensor[Float],
    spatio_scale: Float): Unit = {
    val num_rois = rois.size(1)
    val channels = data.size(2)
    val height = data.size(3)
    val width = data.size(4)

    output.resize(num_rois, channels, pooled_height, pooled_width)
      .fill(ev.fromType[Float](Float.MinValue))
    val output_size = num_rois * channels * pooled_height * pooled_width
    require(output.nElement() != 0, "Output contains no elements")

    for (n <- 1 to num_rois) {
      val roi = rois(n)
      val d = data(roi.valueAt(1).toInt)
      val roi_start_w = roi.valueAt(2) * spatio_scale
      val roi_start_h = roi.valueAt(3) * spatio_scale
      val roi_end_w = roi.valueAt(4) * spatio_scale
      val roi_end_h = roi.valueAt(5) * spatio_scale

      val roi_width = Math.max(roi_end_w - roi_start_w, 1.0f)
      val roi_height = Math.max(roi_end_h - roi_start_h, 1.0f)
      val bin_size_h = roi_height/ pooled_height
      val bin_size_w = roi_width / pooled_width

      val roi_bin_grid_h = if (sampling_ratio > 0) {
        sampling_ratio
      } else {
        Math.ceil(roi_height / pooled_height).toInt
      }

      val roi_bin_grid_w = if (sampling_ratio > 0) {
        sampling_ratio
      } else {
        Math.ceil(roi_width / pooled_width).toInt
      }

      val pre_cal = Tensor[Float](
        Array(pooled_height * pooled_width * roi_bin_grid_h * roi_bin_grid_w, 8))

      val count: Float = roi_bin_grid_h * roi_bin_grid_w

      preCalcForBilinearInterpolateFloat(
        height,
        width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_cal
      )

      for (c <- 1 to channels) {
        var pre_calc_index: Int = 1

        for (ph <- 1 to pooled_height) {
          for (pw <- 1 to pooled_width) {
            var output_val: Float = 0.0f

            for (iy <- 1 to roi_bin_grid_h) {
              for (ix <- 1 to roi_bin_grid_w) {
                val pc = pre_cal(pre_calc_index)
                val x_low = pc.valueAt(1).toInt
                val y_low = pc.valueAt(2).toInt
                val x_high = pc.valueAt(3).toInt
                val y_high = pc.valueAt(4).toInt
                val w1 = pc.valueAt(5)
                val w2 = pc.valueAt(6)
                val w3 = pc.valueAt(7)
                val w4 = pc.valueAt(8)

                output_val +=  w1 * d.valueAt(c, x_low, y_low) +
                  w2 * d.valueAt(c, x_high, y_low) +
                  w3 * d.valueAt(c, x_low, y_high) +
                  w4 * d.valueAt(c, x_high, y_high)

                pre_calc_index += 1
              }
            }
            output_val /= count

            output.setValue(n, c, ph, pw, ev.fromType[Float](output_val))
          }
        }
      }
    }
  }

  private def poolOneRoiDouble(data: Tensor[Double], rois: Tensor[Double],
                              spatio_scale: Double): Unit = {
    val num_rois = rois.size(1)
    val channels = data.size(2)
    val height = data.size(3)
    val width = data.size(4)

    output.resize(num_rois, channels, pooled_height, pooled_width)
      .fill(ev.fromType[Double](Double.MinValue))
    val output_size = num_rois * channels * pooled_height * pooled_width
    require(output.nElement() != 0, "Output contains no elements")

    for (n <- 1 to num_rois) {
      val roi = rois(n)
      val d = data(roi.valueAt(1).toInt)
      val roi_start_w = roi.valueAt(2) * spatio_scale
      val roi_start_h = roi.valueAt(3) * spatio_scale
      val roi_end_w = roi.valueAt(4) * spatio_scale
      val roi_end_h = roi.valueAt(5) * spatio_scale

      val roi_width = Math.max(roi_end_w - roi_start_w, 1.0)
      val roi_height = Math.max(roi_end_h - roi_start_h, 1.0)
      val bin_size_h = roi_height/ pooled_height
      val bin_size_w = roi_width / pooled_width

      val roi_bin_grid_h = if (sampling_ratio > 0) {
        sampling_ratio
      } else {
        Math.ceil(roi_height / pooled_height).toInt
      }

      val roi_bin_grid_w = if (sampling_ratio > 0) {
        sampling_ratio
      } else {
        Math.ceil(roi_width / pooled_width).toInt
      }

      val pre_cal = Tensor[Double](
        Array(pooled_height * pooled_width * roi_bin_grid_h * roi_bin_grid_w, 8))

      val count: Double = roi_bin_grid_h * roi_bin_grid_w

      preCalcForBilinearInterpolateDouble(
        height,
        width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_cal
      )

      for (c <- 1 to channels) {
        var pre_calc_index: Int = 1

        for (ph <- 1 to pooled_height) {
          for (pw <- 1 to pooled_width) {
            var output_val: Double = 0.0

            for (iy <- 1 to roi_bin_grid_h) {
              for (ix <- 1 to roi_bin_grid_w) {
                val pc = pre_cal(pre_calc_index)
                val x_low = pc.valueAt(1).toInt
                val y_low = pc.valueAt(2).toInt
                val x_high = pc.valueAt(3).toInt
                val y_high = pc.valueAt(4).toInt
                val w1 = pc.valueAt(5)
                val w2 = pc.valueAt(6)
                val w3 = pc.valueAt(7)
                val w4 = pc.valueAt(8)

                output_val +=  w1 * d.valueAt(c, x_low, y_low) +
                  w2 * d.valueAt(c, x_high, y_low) +
                  w3 * d.valueAt(c, x_low, y_high) +
                  w4 * d.valueAt(c, x_high, y_high)

                pre_calc_index += 1
              }
            }
            output_val /= count

            output.setValue(n, c, ph, pw, ev.fromType[Double](output_val))
          }
        }
      }
    }
  }

  private def preCalcForBilinearInterpolateFloat(
    height: Int,
    width: Int,
    iy_upper: Int,
    ix_upper: Int,
    roi_start_h: Float,
    roi_start_w: Float,
    bin_size_h: Float,
    bin_size_w: Float,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int,
    pre_cal: Tensor[Float]
  ) : Unit = {
    var pre_calc_index: Int = 1

    for (ph <- 0 until pooled_height) {
      for (pw <- 0 until pooled_width) {
        for (iy <- 0 until iy_upper) {
          val yy = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h
          for (ix <- 0 until ix_upper) {
            val xx = roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w
            var x = xx
            var y = yy
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
              pre_cal.setValue(pre_calc_index, 1, 0.0f) // pos1
              pre_cal.setValue(pre_calc_index, 2, 0.0f) // pos2
              pre_cal.setValue(pre_calc_index, 3, 0.0f) // pos3
              pre_cal.setValue(pre_calc_index, 4, 0.0f) // pos4
              pre_cal.setValue(pre_calc_index, 5, 0.0f) // w1
              pre_cal.setValue(pre_calc_index, 6, 0.0f) // w2
              pre_cal.setValue(pre_calc_index, 7, 0.0f) // w3
              pre_cal.setValue(pre_calc_index, 8, 0.0f) // w4
              pre_calc_index += 1
            }

            else {
              if (y <= 0) {
                y = 0
              }

              if (x <= 0) {
                x = 0
              }

              var y_low = y.toInt
              var x_low = x.toInt

              val y_high = if (y_low >= height - 1) {
                y_low = height -1
                y = y_low.toFloat
                y_low
              } else {
                y_low + 1
              }

              val x_high = if (x_low >= width - 1) {
                x_low = width -1
                x = x_low.toFloat
                x_low
              } else {
                x_low + 1
              }

              val ly = y - y_low
              val lx = x - x_low
              val hy = 1.0f - ly
              val hx = 1.0f - lx
              val w1 = hy * hx
              val w2 = hy * lx
              val w3 = ly * hx
              val w4 = ly * lx

              pre_cal.setValue(pre_calc_index, 1, x_low + 1.0f)
              pre_cal.setValue(pre_calc_index, 2, y_low + 1.0f)
              pre_cal.setValue(pre_calc_index, 3, x_high + 1.0f)
              pre_cal.setValue(pre_calc_index, 4, y_high + 1.0f)
              pre_cal.setValue(pre_calc_index, 5, w1)
              pre_cal.setValue(pre_calc_index, 6, w2)
              pre_cal.setValue(pre_calc_index, 7, w3)
              pre_cal.setValue(pre_calc_index, 8, w4)
              pre_calc_index += 1
            }
          }
        }
      }
    }
  }

  private def preCalcForBilinearInterpolateDouble(
    height: Int,
    width: Int,
    iy_upper: Int,
    ix_upper: Int,
    roi_start_h: Double,
    roi_start_w: Double,
    bin_size_h: Double,
    bin_size_w: Double,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int,
    pre_cal: Tensor[Double]
  ) : Unit = {
    var pre_calc_index = 1

    for (ph <- 0 until pooled_height) {
      for (pw <- 0 until pooled_width) {
        for (iy <- 0 until iy_upper) {
          val yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
          for (ix <- 0 until ix_upper) {
            val xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
            var x = xx
            var y = yy
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
              pre_cal.setValue(pre_calc_index, 1, 0.0) // pos1
              pre_cal.setValue(pre_calc_index, 2, 0.0) // pos2
              pre_cal.setValue(pre_calc_index, 3, 0.0) // pos3
              pre_cal.setValue(pre_calc_index, 4, 0.0) // pos4
              pre_cal.setValue(pre_calc_index, 5, 0.0) // w1
              pre_cal.setValue(pre_calc_index, 6, 0.0) // w2
              pre_cal.setValue(pre_calc_index, 7, 0.0) // w3
              pre_cal.setValue(pre_calc_index, 8, 0.0) // w4
              pre_calc_index += 1
            }

            else {
              if (y <= 0) {
                y = 0
              }

              if (x <= 0) {
                x = 0
              }

              var y_low = y.toInt
              var x_low = x.toInt

              val y_high = if (y_low >= height - 1) {
                y_low = height -1
                y = y_low.toDouble
                y_low
              } else {
                y_low + 1
              }

              val x_high = if (x_low >= width - 1) {
                x_low = width -1
                x = x_low.toDouble
                x_low
              } else {
                x_low + 1
              }

              val ly = y - y_low
              val lx = x - x_low
              val hy = 1.0 - ly
              val hx = 1.0 - lx
              val w1 = hy * hx
              val w2 = hy * lx
              val w3 = ly * hx
              val w4 = ly * lx

              pre_cal.setValue(pre_calc_index, 1, x_low + 1.0)
              pre_cal.setValue(pre_calc_index, 2, y_low + 1.0)
              pre_cal.setValue(pre_calc_index, 3, x_high + 1.0)
              pre_cal.setValue(pre_calc_index, 4, y_high + 1.0)
              pre_cal.setValue(pre_calc_index, 5, w1)
              pre_cal.setValue(pre_calc_index, 6, w2)
              pre_cal.setValue(pre_calc_index, 7, w3)
              pre_cal.setValue(pre_calc_index, 8, w4)
              pre_calc_index += 1
            }
          }
        }
      }
    }
  }
}

object RoiAlign {
  def apply[@specialized(Float, Double) T: ClassTag](
    spatio_scale: T,
    sampling_ratio: Int,
    pooled_height: Int,
    pooled_width: Int) (implicit ev: TensorNumeric[T]): RoiAlign[T] =
    new RoiAlign[T](spatio_scale, sampling_ratio, pooled_height, pooled_width)
}
