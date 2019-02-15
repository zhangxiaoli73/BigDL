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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.{Inception_Layer_v1, Inception_v1_NoAuxClassifier}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.vgg.Vgg_16
import com.intel.analytics.bigdl.nn.StaticGraph
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn.Equivalent
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._

import scala.util.Random

class BlasToDnnSpec extends BigDLSpecHelper {
  "vgg16 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    val gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    val blas = Vgg_16.graph(classNum, false).asInstanceOf[StaticGraph[Float]]
    blas.setInputFormats(Seq(Memory.Format.nchw))
    blas.setOutputFormats(Seq(Memory.Format.nc))
    val irBlas = blas.cloneModule().toIRgraph()

    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputDnn, gradInputBlas, 1e-4) should be(true)
  }

  "lenet5 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 1, 28, 28)

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 10).rand()

    val blas = LeNet5.graph(10).asInstanceOf[StaticGraph[Float]]
    blas.setInputFormats(Seq(Memory.Format.nchw))
    blas.setOutputFormats(Seq(Memory.Format.nc))
    val irBlas = blas.cloneModule().toIRgraph()

    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputDnn, gradInputBlas, 1e-6) should be(true)
  }

  "ResNet-50 blas to dnn" should "work properly" in {
    Engine.setEngineType(MklDnn)
    val batchSize = 4
    val classNum = 1000
    val depth = 50
    RNG.setSeed(1000)
    val input = Tensor[Float](batchSize, 3, 224, 224).rand(-1, 1)
    val gradOutput = Tensor[Float](batchSize, 1000).rand()

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))

    val graphModel = model.toGraph().cloneModule().asInstanceOf[StaticGraph[Float]].toIRgraph()

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null

    for (i <- 1 to 1) {
      model.zeroGradParameters()
      graphModel.zeroGradParameters()
      output1 = graphModel.forward(input).toTensor[Float]
      gradInput1 = graphModel.backward(input, gradOutput).toTensor[Float]

      output2 = model.forward(input).toTensor[Float]
      gradInput2 = model.backward(input, gradOutput).toTensor[Float]
    }

    val (weight1, gradweight1) = model.getParameters()
    val (weight2, gradweight2) = graphModel.getParameters()

    Equivalent.nearequals(weight1, weight2) should be(true)
    Equivalent.nearequals(gradweight1, gradweight2) should be(true)

    Equivalent.nearequals(output1, output2) should be(true)
    Equivalent.nearequals(gradInput1, gradInput2) should be(true)
  }
}
