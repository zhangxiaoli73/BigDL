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

package com.intel.analytics.bigdl.models.rnn

import java.util

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.text.LabeledSentence
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}

import scala.collection.Iterator
import scala.reflect.ClassTag

object BatchPaddingClassify {
  def apply[T: ClassTag]
  (batchSize: Int,
   vocabLength: Int,
   fixDataLength: Option[Int] = None,
   fixLabelLength: Option[Int] = None)
  (implicit ev: TensorNumeric[T]): BatchPaddingClassify[T]
  = new BatchPaddingClassify[T](batchSize, vocabLength, fixDataLength, fixLabelLength)
}

class BatchPaddingClassify[T: ClassTag]
(totalBatch: Int,
 vocabLength: Int,
 fixDataLength: Option[Int] = None,
 fixLabelLength: Option[Int] = None)
(implicit ev: TensorNumeric[T])
  extends Transformer[LabeledSentence[T], MiniBatch[T]] {

  /**
   * only padding feature data, no label data
   * @param prev
   * @return
   */
  override def apply(prev: Iterator[LabeledSentence[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private var featureTensor: Tensor[T] = Tensor[T]()
      private var labelTensor: Tensor[T] = Tensor[T]()
      private var sentenceData: Array[LabeledSentence[T]] = null
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null

      private val batchSize = totalBatch // Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          var maxLength = 0
          val batchLength = new Array[Int](batchSize)
          if (sentenceData == null) sentenceData = new Array[LabeledSentence[T]](batchSize)
          while (i < batchSize && prev.hasNext) {
            val sentence = prev.next()
            val dataLength = sentence.dataLength()
            sentenceData(i) = sentence
            // update length
            if (dataLength > maxLength) maxLength = dataLength
            batchLength(i) = dataLength
            i += 1
          }

          val dataLength = fixDataLength.getOrElse(maxLength)
          val labelLength = fixLabelLength.getOrElse(maxLength)

          if (featureData == null || featureData.length < dataLength * vocabLength) {
            featureData = new Array[T](batchSize * dataLength * vocabLength)
          }
          if (labelData == null || labelData.length < labelLength) {
            labelData = new Array[T](batchSize * labelLength)
          }
          featureSize = Array(i, maxLength, vocabLength)
          labelSize = Array(i, maxLength)
          // init
          ev.getType() match {
            case DoubleType =>
              util.Arrays.fill(
                featureData.asInstanceOf[Array[Double]], 0, featureData.length, 0.0)
              util.Arrays.fill(labelData.asInstanceOf[Array[Double]], 0, labelData.length, 0.0)
            case FloatType =>
              util.Arrays.fill(
                featureData.asInstanceOf[Array[Float]], 0, featureData.length, 0.0f)
              util.Arrays.fill(labelData.asInstanceOf[Array[Float]], 0, labelData.length, 0.0f)
            case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
          }

          // padding
          i = 0
          while (i < batchLength.length) {
            val sentence = sentenceData(i)
            val startTokenIndex = sentence.getData(0)
            val endTokenIndex = if (labelLength == 1) 0
            else ev.toType[Int](sentence.getLabel(sentence.labelLength - 1))

            var j = 0
            while (j < sentence.dataLength) {
              featureData(i * maxLength + j * vocabLength + ev.toType[Int](sentence.getData(j)))
                = ev.fromType[Float](1.0f)
              labelData(i * maxLength + j) = ev.plus(sentence.label()(j), ev.fromType[Float](1.0f))
              j += 1
            }
            while (j < maxLength) {
              featureData(i * maxLength + j * vocabLength + endTokenIndex) =
                ev.fromType[Float](1.0f)
              j += 1
            }
            i += 1
          }

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