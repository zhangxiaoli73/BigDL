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

package com.intel.analytics.bigdl.models.maskrcnn

import com.intel.analytics.bigdl.models.resnet.Utils.{TestParams, _}
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.T
import scopt.OptionParser

object Test {
  case class TestParams(
                         folder: String = "./",
                         model: String = "",
                         batchSize: Int = 128
                       )

  val testParser = new OptionParser[TestParams]("BigDL ResNet on Cifar10 Test Example") {
    opt[String]('f', "folder")
      .text("the location of Cifar10 dataset")
      .action((x, c) => c.copy(folder = x))

    opt[String]('m', "model")
      .text("the location of model snapshot")
      .action((x, c) => c.copy(model = x))
      .required()
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach { param => {
      val dummyInput = T()

      val trans = BytesToMat() -> Resize(minSize, maxSize) ->
        ChannelNormalize(123f, 115f, 102.9801f) -> MatToTensor[Float]()

    }}
  }
}
