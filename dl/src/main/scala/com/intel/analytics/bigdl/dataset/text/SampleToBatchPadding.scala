/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset.text

import java.util

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}

import scala.collection.Iterator
import scala.reflect.ClassTag

object SampleToBatchPadding {
  def apply[T: ClassTag]
  (totalBatch : Int,
   padding : Boolean = false,
   padValue : Option[T] = None,
   fixLength : Option[Int] = None)
  (implicit ev: TensorNumeric[T]): SampleToBatchPadding[T]
  = new SampleToBatchPadding[T](totalBatch, padding, padValue, fixLength)
}

class SampleToBatchPadding[T: ClassTag]
(totalBatch : Int,
 padding : Boolean,
 paddingValue : Option[T] = None,
 fixLength : Option[Int] = None)
(implicit ev: TensorNumeric[T])
  extends Transformer[Sample[T], MiniBatch[T]] {

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private var featureTensor: Tensor[T] = Tensor[T]()
      private var labelTensor: Tensor[T] = Tensor[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null

      private val batchSize = Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          val padValue = paddingValue.getOrElse(ev.zero)
          val startToken = 0
          val endToken = 4
          var maxLength = 0
          var i = 0
          var maxSize = new Array[Int](2)
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            val feature = sample.feature()
            val label = sample.label()
            require(feature.isContiguous() && label.isContiguous(), "SampleToBatchPadding: " +
              "Only support contiguous tensor, pls use tensor.contiguous() before batching")
            // require(feature.size(1) == label.size(1), "SampleToBatchPadding: Padding label")
            if (i == 0) maxSize = feature.size()
            if (featureData == null || featureData.length < batchSize * maxSize.product) {
              featureData = new Array[T](batchSize * maxSize.product)
            }
            if (labelData == null || labelData.length < batchSize * maxSize(0)) {
              labelData = new Array[T](batchSize * maxSize(0))
            }
            if (i == 0) {
              ev.getType() match {
                case DoubleType =>
                  util.Arrays.fill(
                    featureData.asInstanceOf[Array[Double]], 0, featureData.length, 0.0)
                  util.Arrays.fill(
                    featureData.asInstanceOf[Array[Double]], 0, batchSize * maxSize.product,
                    ev.toType[Double](padValue))
                  util.Arrays.fill(
                    labelData.asInstanceOf[Array[Double]], 0, labelData.length, 0.0)
                  util.Arrays.fill(
                    labelData.asInstanceOf[Array[Double]], 0, batchSize * maxSize(0),
                    ev.toType[Double](padValue))
                case FloatType =>
                  util.Arrays.fill(
                    featureData.asInstanceOf[Array[Float]], 0, featureData.length, 0.0f)
                  util.Arrays.fill(
                    featureData.asInstanceOf[Array[Float]], 0, batchSize * maxSize.product,
                    ev.toType[Float](padValue))
                  util.Arrays.fill(
                    labelData.asInstanceOf[Array[Float]], 0, labelData.length, 0.0f)
                  util.Arrays.fill(
                    labelData.asInstanceOf[Array[Float]], 0, batchSize * maxSize(0),
                    ev.toType[Float](padValue))
                case _ => throw new
                    UnsupportedOperationException(s"BatchPaddingLM: Only Float/Double supported")
              }
            }

            val oneLabelLength = sample.label().nElement()
            val oneFeatureLength = sample.feature().nElement()
            sample.copyFromLabel(labelData, i * maxSize(0), oneLabelLength)
            sample.copyFromFeature(featureData, i * maxSize.product, oneFeatureLength)

            var j = oneLabelLength
            while (j < maxSize(0)) {
              labelData(i * maxSize(0) + j) = ev.plus(ev.fromType(startToken), ev.one)
              featureData(i * maxSize.product + j * maxSize(1) + endToken) =
                ev.fromType[Float](1.0f)
              j += 1
            }

            i += 1
          }
          featureSize = Array(i) ++ maxSize
          labelSize = Array(i, maxSize(0))

          featureTensor.set(Storage[T](featureData),
            storageOffset = 1, sizes = featureSize)
          labelTensor.set(Storage[T](labelData),
            storageOffset = 1, sizes = labelSize)
          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}
