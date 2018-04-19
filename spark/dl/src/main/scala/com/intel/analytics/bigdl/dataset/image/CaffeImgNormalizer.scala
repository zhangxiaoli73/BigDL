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

import com.intel.analytics.bigdl.dataset.{LocalDataSet, Transformer}
import org.apache.log4j.Logger

import scala.collection.Iterator

object CaffeImgNormalizer {
  val logger = Logger.getLogger(getClass)

  def apply(meanR: Int, meanG: Int, meanB: Int, scale: Double): CaffeImgNormalizer = {
    new CaffeImgNormalizer(meanR, meanG, meanB, scale)
  }

  class CaffeImgNormalizer(meanR: Int, meanG: Int, meanB: Int, scale: Double)
    extends Transformer[LabeledBGRImage, LabeledBGRImage] {

    private var buffer : LabeledBGRImage = null

    def getMean(): (Int, Int, Int) = (meanR, meanG, meanB)

    override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
      prev.map(img => {
        val content = img.content
        require(content.length % 3 == 0)
        var i = 0
        val frameLength = content.length / 3
        val height = img.height()
        val width = img.width()

        if (buffer == null) buffer = new LabeledBGRImage(width, height)
        val bufferContent = buffer.content

        val channels = 3
        val mean = Array(meanR, meanG, meanB)
        var c = 0
        while (c < channels) {
           i = 0
          while (i < frameLength) {
//            val th = i / width
//            val tw = i % width
//            val top_index = if (mirror) {
//              c * frameLength + i + width - 2 * tw - 1
//            } else {
//              c * frameLength + i
//            }
            val data_index = c * frameLength + i
            bufferContent(data_index) = ((content(data_index) - mean(c)) * scale).toFloat
            i += 1
          }
          c += 1
        }
        buffer.setLabel(img.label())
        buffer
      })
    }
  }
}


