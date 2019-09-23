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

import breeze.linalg.{max, min}
import breeze.numerics.round
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Resize
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, augmentation}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Resize._
import com.intel.analytics.bigdl.transform.vision.image.label.roi.{RoiLabel, RoiResize}
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import org.apache.log4j.Logger
import org.apache.spark.ml
import org.apache.spark.ml.feature
import org.apache.spark.util.random
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class ResizeMask(minSize: Int = -1, maxSize: Int = -1)
  extends FeatureTransformer {
  private def getSize(sizeH: Int, sizeW: Int): (Int, Int) = {
    var size = minSize
    if (maxSize > 0) {
      val min_original_size = math.min(sizeW, sizeH)
      val max_original_size = math.max(sizeW, sizeH)
      if (max_original_size / min_original_size * size > maxSize) {
        size = math.round(maxSize * min_original_size / max_original_size)
      }
    }

    if ((sizeW <= sizeH && sizeW == size) || (sizeH <= sizeW && sizeH == size)) {
      return (sizeH, sizeW)
    }

    if (sizeW < sizeH) {
      return (size * sizeH / sizeW, size)
    } else {
      return (size, size * sizeW / sizeH)
    }
  }

  override def transformMat(feature: ImageFeature): Unit = {
    val sizes = this.getSize(feature.getHeight(), feature.getWidth())

    val resizeH = sizes._1
    val resizeW = sizes._2

    Imgproc.resize(feature.opencvMat(), feature.opencvMat(), new Size(resizeW, resizeH))

    // resize roi label
    if (feature.hasLabel()) {
      // bbox resize
      transformBbox(feature)
      // mask resize
      transformMaskPoly(feature)
    }
  }

  private def transformBbox(feature: ImageFeature): Unit = {
    val scaledW = feature.getWidth().toFloat / feature.getOriginalWidth
    val scaledH = feature.getHeight().toFloat / feature.getOriginalHeight
    val target = feature.getLabel[RoiLabel]
    BboxUtil.scaleBBox(target.bboxes, scaledH, scaledW)
  }

  private def transformMaskPoly(feature: ImageFeature): Unit = {
    val scaledW = feature.getWidth().toFloat / feature.getOriginalWidth
    val scaledH = feature.getHeight().toFloat / feature.getOriginalHeight

    val mask = feature.getLabel[RoiLabel].mask
    // mask shape (poly number, 2)
    if (mask != null) { // not mask
      mask.select(2, 1).mul(scaledW)
      mask.select(2, 2).mul(scaledH)
    }
  }
}

object ResizeMask {
  val logger = Logger.getLogger(getClass)

  def apply(minSize: Int = -1, maxSize: Int = -1): ResizeMask = new ResizeMask(minSize, maxSize)
}