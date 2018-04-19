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


import java.nio.ByteBuffer

import breeze.linalg.normalize
import com.intel.analytics.bigdl.dataset.{ByteRecord, Transformer}
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat

import scala.collection.Iterator


object BytesToMat {
  def apply(): BytesToMat =
    new BytesToMat()
}

class BytesToMat() extends Transformer[ByteRecord, ImageFeature] {
  override def apply(prev: Iterator[ByteRecord]): Iterator[ImageFeature] = {
    prev.map(byteRecord => {
      val rawData = byteRecord.data
      val imgBuffer = ByteBuffer.wrap(rawData)
      val width = imgBuffer.getInt
      val height = imgBuffer.getInt
      val bytes = new Array[Byte](3 * width * height)
      // System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
      copySrc(imgBuffer.array(), bytes, 8)
      // val mat = OpenCVMat.pixelsBytesToMat(bytes, height, width)
      val mat = OpenCVMat.fromPixelsBytes(bytes, height, width, 3)
      val feature = new ImageFeature()
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalSize) = mat.shape()
      feature(ImageFeature.label) = byteRecord.label
      feature
    })
  }

  def copySrc(rawData: Array[Byte], dstData: Array[Byte], offset: Int): Unit = {
    val buffer = ByteBuffer.wrap(rawData)
    val _width = buffer.getInt
    val _height = buffer.getInt
    require(rawData.length == offset + _width * _height * 3)
    val frameLength = _width * _height
    var j = 0
    while (j < frameLength) {
      dstData(j) = rawData(j*3 + offset)
      dstData(j + frameLength) = rawData(j*3 + offset + 1)
      dstData(j + frameLength * 2) = rawData(j*3 + offset + 2)
      j += 1
    }
  }
}


