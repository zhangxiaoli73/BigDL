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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.example.loadmodel.AlexNet_OWT
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Tensor, Storage}
import com.intel.analytics.bigdl.utils.{File, ShareGradInput, T, Table}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import scala.util.Random

class SharedGradInputSpec extends FlatSpec with Matchers {


  "Inception+ShareGradInput" should "generate correct output" in {
    Random.setSeed(3)
    val input = Tensor[Float](4, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => Random.nextInt(1000))

    val seed = 100
    RNG.setSeed(seed)

    val model = Inception.getModel[Float](1000, "inception-bn")
    val sharedModel = Inception.getModel[Float](1000, "inception-bn")
    ShareGradInput.shareConvolution(sharedModel)
    ShareGradInput.shareGradInput(sharedModel)

    val (weights, grad) = model.getParameters()
    val (w, g) = sharedModel.getParameters()
    w.copy(weights)

    val output1 = model.forward(input).toTensor[Float]
    val output2 = sharedModel.forward(input).toTensor[Float]
    output1 should be (output2)
    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)
    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = sharedModel.backward(input, gradOutput)
    gradInput1 should be (gradInput2)

    File.save(model, "/tmp/unshared.inception", true)
    File.save(sharedModel, "/tmp/shared.inception2", true)

    println()

  }

  "ResNet+ShareGradInput" should "generate correct output" in {
    val inputSeed = 1
    val depth = 18
    val batchSize = 4
    val modelSeed = 101
    Random.setSeed(inputSeed)
    val classNum: Int = 1000
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
    val labels = Tensor[Float](batchSize).apply1(e => Random.nextInt(classNum))
    val seed = modelSeed
    RNG.setSeed(seed)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    val sharedModel = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    ShareGradInput.shareConvolution(sharedModel)
    ShareGradInput.shareGradInput(sharedModel)

    val (weights, grad) = model.getParameters()
    val (w, g) = sharedModel.getParameters()
    w.copy(weights)

    val output1 = model.forward(input).toTensor[Float]
    val output2 = sharedModel.forward(input).toTensor[Float]
    output1 should be (output2)
    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)
    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = sharedModel.backward(input, gradOutput)
    gradInput1 should be (gradInput2)

    File.save(model, "/tmp/unshared.resnet", true)
    File.save(sharedModel, "/tmp/shared.resnet", true)

    println()
  }

  "AlexNet+ShareGradInput" should "generate correct output" in {
    Random.setSeed(1)
    val input = Tensor[Float](8, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](8).apply1(e => Random.nextInt(100))

    val seed = 100
    RNG.setSeed(seed)
    val model = AlexNet_OWT(1000, false, true)
    val sharedModel = AlexNet_OWT(1000, false, true)
    ShareGradInput.shareConvolution[Float](sharedModel)
    ShareGradInput.shareGradInput[Float](sharedModel)

    val (weights, grad) = model.getParameters()
    val (w, g) = sharedModel.getParameters()
    w.copy(weights)

    val output1 = model.forward(input).toTensor[Float]
    val output2 = sharedModel.forward(input).toTensor[Float]
    output1 should be (output2)
    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)
    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = sharedModel.backward(input, gradOutput)
    gradInput1 should be (gradInput2)

    File.save(model, "/tmp/unshared.alexnet", true)
    File.save(sharedModel, "/tmp/shared.alexnet", true)
  }
}
