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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.dataset.text.LabeledSentenceToSample
import com.intel.analytics.bigdl.models.rnn.Utils._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class BatchPaddingSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "BatchPadding with Float Array input and Array label" should "be good in copyToData" +
    "and copyToLabel" in {

    val folder = "/home/zhangli/CodeSpace/forTrain/rnn/test"
    val dictionaryLength = 4001
    val wt = new WordTokenizer(folder + "/input1.txt",
      folder, dictionaryLength = dictionaryLength)
    wt.process()

    val dataArray = loadInData(folder, dictionaryLength)
    val trainData = dataArray._1
    val valData = dataArray._2
    val trainMaxLength = dataArray._3
    val valMaxLegnth = dataArray._4

    val batchSize = 1

    val trainData1 = trainData.sortBy(_.labelLength())
    val valData1 = valData.sortBy(_.labelLength())

    val trainSet1 = DataSet.array(trainData1)
      .transform(LabeledSentenceToSample(dictionaryLength,
        Some(trainMaxLength), Some(trainMaxLength)))
      .transform(SampleToBatch(batchSize = batchSize))

    /*
    val trainSet2 = DataSet.array(trainData1)
      .transform(BatchPaddingLM(batchSize = batchSize, dictionaryLength,
        Some(trainMaxLength), Some(trainMaxLength)))

    val data1 = trainSet1.toLocal().data(train = false)
    val data2 = trainSet2.toLocal().data(train = false)
    while (data1.hasNext && data2.hasNext) {
      val batch1 = data1.next()
      val input1 = batch1.data
      val label1 = batch1.labels

      val batch2 = data2.next()
      val input2 = batch2.data
      val label2 = batch2.labels
      val length = label2.size(2)

      val arrLabel1 = input1.storage()
      val arrLabel2 = input2.storage()

      val arr1 = input1.storage()
      val arr2 = input2.storage()

      arr1.length() should be(arr2.length())
      var i = 0
      while (i < dictionaryLength * length) {
        arr1(i) should be (arr2(i))
        i += 1
      }

      i = 0
      while (i < length) {
        arrLabel1(i) should be (arrLabel2(i))
        i += 1
      }
    }
    while (data1.hasNext) {
      println("dataset to sample")
    }
    while (data2.hasNext) {
      println("dataset to batch padding")
    }
    */
  }
}
