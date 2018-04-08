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
package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v1_NoAuxClassifier, Inception_v2}
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType
import com.intel.analytics.bigdl.models.vgg.{VggForCifar10, Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class VggModelSpec extends FlatSpec with Matchers {

  def getModel(module: String, batchSize: Int): (Module[Float], MiniBatch[Float]) = {
    RNG.setSeed(100)
    val (_model, input) = module match {
      case "inception_v1" =>
        (Inception_v1(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "inception_v1_dnn" =>
        (Inception_v1_dnn(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "inception_v2" =>
        (Inception_v2(1000), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "inception_v2_dnn" =>
        (Inception_v2_dnn(1000), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 3000).randn()))
      case "inception_no_dnn" =>
        (Inception_v1_NoAuxClassifier_dnn(1000, false),
          MiniBatch(Tensor[Float](batchSize, 3, 224, 224).randn(),
            Tensor[Float](batchSize).fill(1)))
      case "inception_no" =>
        (Inception_v1_NoAuxClassifier(1000, false),
          MiniBatch(Tensor[Float](batchSize, 3, 224, 224).randn(),
            Tensor[Float](batchSize).fill(1)))
      case "vgg16" =>
        (Vgg_16(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "vgg16_dnn" =>
        (Vgg_16_dnn(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "vgg19" =>
        (Vgg_19(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "vgg19_dnn" =>
        (Vgg_19_dnn(1000, false), MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
      case "resnet_50" =>
        val model = ResNet(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataSet" -> DatasetType.ImageNet))
//        ResNet.shareGradInput(model)
//        ResNet.modelInit(model)
        (model, MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))

      case "resnet_50_dnn" =>
        val model = ResNet_dnn(classNum = 1000, T("depth" -> 50, "optnet" -> true,
          "dataSet" -> ResNet_dnn.DatasetType.ImageNet))
        //        ResNet_dnn.shareGradInput(model)
//        ResNet_dnn.modelInit(model)
        (model, MiniBatch(Tensor[Float](batchSize, 3, 224, 224)
          .apply1(e => RNG.uniform(0, 1).toFloat), Tensor[Float](batchSize, 1000).randn()))
    }
    _model.createDnnEngine(0)
    _model.createStream()
    (_model, input)
  }

  def formatEqual(out1: Tensor[Float], out2: Tensor[Float],
                  epsilion: Double, format: Int = 0): Unit = {
    var userOut2 = Tensor[Float]()
    if (out1.getFormat() != out2.getFormat() && out2.getFormat() != 5) {
      DnnUtils.reorderToUser(out2, userOut2, 5)
    } else {
      userOut2 = out2
    }

    DnnUtils.nearequals(out1, userOut2, epsilion) should be(true)
  }

  "VGG on Cifar10" should "work correctly" in {
    RNG.setSeed(1)
    val dnn = VggForCifar10.dnn(10, hasDropout = false)
    RNG.setSeed(1)
    val blas = VggForCifar10(10, hasDropout = false)

    dnn.getParameters()._1.copy(blas.getParameters()._1)
    dnn.getParameters()._1 should be (blas.getParameters()._1)

    RNG.setSeed(10)
    val input = Tensor[Float](4, 3, 32, 32).apply1(e => RNG.uniform(-1, 1).toFloat)
    val gradOutput = Tensor[Float]()

    for (i <- 0  to 0) {
      println("round " + i)
      DnnUtils.getunequals(blas.getParameters()._1, dnn.getParameters()._1, 1e-10) should be (true)
      val out1 = blas.forward(input).toTensor[Float]
      val out2 = dnn.forward(input).toTensor[Float]

      println("forward done")
      var userOut2 = Tensor[Float]()
      if (out1.dim() > 2 && out1.getFormat() != out2.getFormat() && out2.getFormat() != 5 && out2.getFormat() != 4) {
        DnnUtils.reorderToUser(out2, userOut2, 5)
      } else {
        userOut2 = out2
      }
      DnnUtils.nearequals(out1, userOut2, 1e-3) should be (true)

      gradOutput.resizeAs(blas.output.toTensor[Float]).copy(blas.output.toTensor[Float])
      
      println("backward start")
      val grad1 = blas.backward(input, gradOutput).toTensor[Float]
      val grad2 = dnn.backward(input, gradOutput).toTensor[Float]

      println("backward done")
      var userGrad2 = Tensor[Float]()
      if (grad1.dim() > 2 & grad1.getFormat() != grad2.getFormat() && grad2.getFormat() != 5 && grad2.getFormat() != 4) {
        DnnUtils.reorderToUser(grad2, userGrad2, 5)
      } else {
        userGrad2 = grad2
      }
      DnnUtils.nearequals(grad1, userGrad2, 1e-3) should be (true)
    }

    println("compare weight ")
    DnnUtils.getunequals(blas.getParameters()._1, dnn.getParameters()._1, 1e-10) should be (true)
    println("compare gradweight ")
    DnnUtils.getunequals(blas.getParameters()._2, dnn.getParameters()._2, 1e-3) should be (true)

    println("done")
  }
}
