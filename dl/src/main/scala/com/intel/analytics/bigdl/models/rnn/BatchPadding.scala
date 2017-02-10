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
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object BatchPadding {
  def apply[T: ClassTag]
  (batchSize: Int)
  (implicit ev: TensorNumeric[T]): BatchPadding[T]
  = new BatchPadding[T](batchSize)
}

class BatchPadding[T: ClassTag]
(totalBatch: Int)
(implicit ev: TensorNumeric[T])
  extends Transformer[LabeledSentence[T], MiniBatch[T]] {

  override def apply(prev: Iterator[LabeledSentence[T]]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {
      private var featureTensor: Tensor[T] = Tensor[T]()
      private var labelTensor: Tensor[T] = Tensor[T]()
      private var featureDataOri = new ArrayBuffer[T]()
      private var labelDataOri = new ArrayBuffer[T]()
      private var featureData: Array[T] = null
      private var labelData: Array[T] = null

      private val batchSize = totalBatch // Utils.getBatchSize(totalBatch)
      private var featureSize: Array[Int] = null
      private var labelSize: Array[Int] = null
      private var oneFeatureLength: Int = 0
      private var oneLabelLength: Int = 0
      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          var vocabLength = 4001
          var maxLength = 0
          var batchLength = new Array[Int](batchSize)
          while (i < batchSize && prev.hasNext) {
            val sentence = prev.next()
            val dataLength = sentence.dataLength()
            val labelLength = sentence.labelLength()

            var m = 0
            while (m < sentence.dataLength) {
              featureDataOri.append(sentence.getData(m))
              m += 1
            }
            m = 0
            while (m < sentence.labelLength) {
              labelDataOri.append(ev.plus(sentence.label()(m), ev.fromType(1)))
              m += 1
            }
            // update length
            if (labelLength > maxLength) maxLength = labelLength

            batchLength(i) = labelLength
            i += 1
          }
          featureSize = Array(i, maxLength, vocabLength)
          labelSize = Array(i, maxLength)
          featureData = new Array[T](featureSize.product)
          labelData = new Array[T](labelSize.product)
          // padding
          ev.getType() match {
            case DoubleType =>
              util.Arrays.fill(
                featureData.asInstanceOf[Array[Double]], 0, featureData.length, 0.0)
            case FloatType =>
              util.Arrays.fill(
                featureData.asInstanceOf[Array[Float]], 0, featureData.length, 0.0f)
            case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
          }

          i = 0
          var nn = 0
          while (i < batchLength.length) {
            var j = 0 // batchLength(i)
            while (j < batchLength(i)) {
              featureData(i * maxLength * vocabLength + j * vocabLength +
                ev.toType[Int](featureDataOri.remove(0))) = ev.fromType[Float](1.0f)
              nn += 1
              j += 1
            }
            i += 1
          }

          i = 0
          while (i < batchLength.length) {
            var j = 0
            while (j < batchLength(i)) {
              labelData(i * maxLength + j) = labelDataOri.remove(0)
               j += 1
            }
            // System.arraycopy(labelDataOri.toArray, i * batchLength(i), labelData,
            //  i * maxLength, batchLength(i))
            util.Arrays.fill(labelData.asInstanceOf[Array[Float]],
              (i) * maxLength + batchLength(i), (i + 1) * maxLength, 1.0f)
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

object LabeledSentenceToSample11 {
  def apply[T: ClassTag]
  (vocabLength: Int,
   fixDataLength: Option[Int] = None,
   fixLabelLength: Option[Int] = None)
  (implicit ev: TensorNumeric[T])
  : LabeledSentenceToSample11[T] =
    new LabeledSentenceToSample11[T](
      vocabLength,
      fixDataLength,
      fixLabelLength)
}

class LabeledSentenceToSample11[T: ClassTag](vocabLength: Int,
                                           fixDataLength: Option[Int],
                                           fixLabelLength: Option[Int])
                                          (implicit ev: TensorNumeric[T])
  extends Transformer[LabeledSentence[T], Sample[T]] {
  private val buffer = Sample[T]()
  private var featureBuffer: Array[T] = null
  private var labelBuffer: Array[T] = null

  override def apply(prev: Iterator[LabeledSentence[T]]): Iterator[Sample[T]] = {
    prev.map(sentence => {

      val dataLength = fixDataLength.getOrElse(sentence.dataLength())
      val labelLength = fixLabelLength.getOrElse(sentence.labelLength())

      if (featureBuffer == null || featureBuffer.length < dataLength * vocabLength) {
        featureBuffer = new Array[T](dataLength * vocabLength)
      }
      if (labelBuffer == null || labelBuffer.length < labelLength) {
        labelBuffer = new Array[T](labelLength)
      }

      // initialize featureBuffer to 0.0

      ev.getType() match {
        case DoubleType =>
          util.Arrays.fill(featureBuffer.asInstanceOf[Array[Double]], 0, featureBuffer.length, 0.0)
          util.Arrays.fill(labelBuffer.asInstanceOf[Array[Double]], 0, labelBuffer.length, 0.0)
        case FloatType =>
          util.Arrays.fill(featureBuffer.asInstanceOf[Array[Float]], 0, featureBuffer.length, 0.0f)
          util.Arrays.fill(labelBuffer.asInstanceOf[Array[Float]], 0, labelBuffer.length, 0.0f)
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }

      /* One-Hot format for feature
       * Expected transformed format should be:
       *
       * Example1: Input = [0, 2, 3], label = [2, 3, 1], dictionary length = 4
       * Transformed: Input = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
       * Transformed: label = [3, 4, 2] (+1 because Tensor index starts from 1)
       *
       * Example2: Input = [0, 2, 3], label = [0], dictionary length = 4
       * Transformed: Input = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
       * Transformed: label = [1] (+1 because Tensor index starts from 1)
       */

      val startTokenIndex = sentence.getData(0)
      val endTokenIndex = if (labelLength == 1) 0
      else ev.toType[Int](sentence.getLabel(sentence.labelLength - 1))

      var i = 0
      while (i < sentence.dataLength) {
        featureBuffer(i*vocabLength + ev.toType[Int](sentence.getData(i)))
          = ev.fromType[Float](1.0f)
        i += 1
      }
      while (i < dataLength) {
        featureBuffer(i*vocabLength + endTokenIndex) = ev.fromType[Float](1.0f)
        i += 1
      }

      i = 0
      while (i < sentence.labelLength) {
        labelBuffer(i) = ev.plus(sentence.label()(i), ev.fromType[Float](1.0f))
        i += 1
      }
      while (i < labelLength) {
        labelBuffer(i) = ev.plus(startTokenIndex, ev.fromType[Float](1.0f))
        i += 1
      }

      buffer.set(featureBuffer, labelBuffer,
        Array(dataLength, vocabLength), Array(labelLength))
    })
  }
}