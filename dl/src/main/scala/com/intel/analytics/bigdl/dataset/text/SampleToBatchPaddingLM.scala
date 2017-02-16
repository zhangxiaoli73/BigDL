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

object SampleToBatchPaddingLM {
  def apply[T: ClassTag]
  (totalBatch : Int,
   startToken: Int,
   endToken: Int,
   padding : Boolean = false,
   padValue : Option[T] = None)
  (implicit ev: TensorNumeric[T]): SampleToBatchPaddingLM[T]
  = new SampleToBatchPaddingLM[T](totalBatch, startToken, endToken, padding, padValue)
}

class SampleToBatchPaddingLM[T: ClassTag]
(totalBatch : Int,
 startToken: Int,
 endToken: Int,
 padding : Boolean,
 paddingValue : Option[T] = None)
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
      private var oneFeatureLength: Int = 0
      private var oneLabelLength: Int = 0
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          val padValue = paddingValue.getOrElse(ev.zero)
          var i = 0
          var maxDataSize: Array[Int] = null
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            require(sample.feature().isContiguous() && sample.label().isContiguous(),
              "SampleToBatchPaddingLM: Only support contiguous tensor, pls use " +
                "tensor.contiguous() before batching")
            require(sample.feature().size(1) == sample.label().size(1),
              "SampleToBatchPaddingLM: first dimension for feature and label should " +
                "be same in language model")

            if (i == 0) {
              maxDataSize = sample.feature().size()
              oneFeatureLength = sample.feature().nElement()
              oneLabelLength = sample.label().nElement()
            }
            if (featureData == null || featureData.length < batchSize * maxDataSize.product) {
              featureData = new Array[T](batchSize * maxDataSize.product)
            }
            if (labelData == null || labelData.length < batchSize * maxDataSize(0)) {
              labelData = new Array[T](batchSize * maxDataSize(0))
            }
            if (i == 0) {
              ev.getType() match {
                case DoubleType =>
                  util.Arrays.fill(featureData.asInstanceOf[Array[Double]], 0, featureData.length,
                    ev.toType[Double](padValue))
                  util.Arrays.fill(labelData.asInstanceOf[Array[Double]], 0, labelData.length,
                    ev.toType[Double](padValue))
                case FloatType =>
                  util.Arrays.fill(featureData.asInstanceOf[Array[Float]], 0, featureData.length,
                    ev.toType[Float](padValue))
                  util.Arrays.fill(labelData.asInstanceOf[Array[Float]], 0, labelData.length,
                    ev.toType[Float](padValue))
                case _ => throw new UnsupportedOperationException(
                  "SampleToBatchPaddingLM: Only Float/Double supported")
              }
            }

            sample.copyFromLabel(labelData, i * oneLabelLength, sample.label().nElement())
            sample.copyFromFeature(featureData, i * oneFeatureLength, sample.feature().nElement())

            var j = sample.label().size(1)
            while (j < maxDataSize(0)) {
              featureData(i * maxDataSize.product + j * maxDataSize(1) + endToken) = ev.one
              labelData(i * maxDataSize(0) + j) = ev.plus(ev.fromType(startToken), ev.one)
              j += 1
            }

            i += 1
          }
          featureSize = Array(i) ++ maxDataSize
          labelSize = Array(i, maxDataSize(0))

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