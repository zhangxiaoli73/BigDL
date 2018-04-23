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

import breeze.numerics.sqrt
import org.opencv.core.{CvType, Mat, Rect}
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.opencv.OpenCV
import org.opencv.imgproc.Imgproc

import scala.collection.Iterator
import com.intel.analytics.bigdl.opencv
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Crop
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.apache.spark.ml
import org.apache.spark.ml.feature
import org.opencv.core.Size

object CaffeImgRandomAspect {
  def apply(min_area_ratio: Float = 0.08f,
            max_area_ratio: Int = 1,
            min_aspect_ratio_change: Float = 0.75f,
            interp_mode: String = "CUBIC"): CaffeImgRandomAspect = {
    OpenCV.isOpenCVLoaded
    new CaffeImgRandomAspect(min_area_ratio, max_area_ratio, min_aspect_ratio_change, interp_mode)
  }
}

// if training, should be random crop, otherwise center crop
class CaffeImgRandomAspect(min_area_ratio: Float = 0.08f,
                           max_area_ratio: Int = 1,
                           min_aspect_ratio_change: Float = 0.75f,
                           interp_mode: String = "CUBIC")
  extends Transformer[ImageFeature, ImageFeature] {

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer: LabeledBGRImage = null

  private val cropLength: Int = 224

  def randRatio(min: Float, max: Float): Float = {
    // return (rand_num(int((max - min) * 1000 + 1)) + min * 1000) / 1000;
    val res = (RNG.uniform(1e-2, (max - min) * 1000 + 1) + min * 1000) / 1000
    res.toFloat
  }

  def resizeImagePerShorterSize(img: Mat, shorter_size: Int) : (Int, Int) = {
    val h = img.size().height
    val w = img.size().width
    var new_h = shorter_size
    var new_w = shorter_size

    if (h < w) {
      new_w = (w / h * shorter_size).toInt
    } else {
      new_h = (h / w * shorter_size).toInt
    }
    (new_h, new_w)
  }

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = prev.map(img => {
    val h = img.opencvMat().size().height
    val w = img.opencvMat().size().width
    val area = h * w
    require(min_area_ratio <= max_area_ratio, "min_area_ratio should <= max_area_ratio")

    var attempt = 0
    while (attempt < 10) {
      val area_ratio = randRatio(min_area_ratio, max_area_ratio)
      val aspect_ratio_change = randRatio(min_aspect_ratio_change, 1 / min_aspect_ratio_change)
      val new_area = area_ratio * area
      var new_h = (sqrt(new_area) * aspect_ratio_change).toInt
      var new_w = (sqrt(new_area) / aspect_ratio_change).toInt
      if (randRatio(0, 1) < 0.5) {
        val tmp = new_h
        new_h = new_w
        new_w = tmp
      }

//      new_h = 404
//      new_w = 605
      if (new_h <= h && new_w <= w) {
        val y = RNG.uniform(1e-2, h - new_h + 1).toInt
        val x = RNG.uniform(1e-2, w - new_w + 1).toInt
        Crop.transform(img.opencvMat(), img.opencvMat(), x, y, x + new_w, y + new_h, false, false)
        Imgproc.resize(img.opencvMat(), img.opencvMat(), new Size(cropLength, cropLength), 0, 0, 2)

//       val croppedImg = new Mat(img.opencvMat(), new Rect(x, y, new_w, new_h))
//       conresponding to interp_mode = cv::INTER_CUBIC;
        // img(ImageFeature.mat) = croppedImg
        attempt = 100
      }
      attempt += 1
    }
    if (attempt < 20) {
      val (new_h, new_w) = resizeImagePerShorterSize(img.opencvMat(), cropLength)
      Imgproc.resize(img.opencvMat(), img.opencvMat(), new Size(cropLength, cropLength), 0, 0, 2)
    }
    img
  })
}

