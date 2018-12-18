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


package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.mkldnn.Equivalent
import com.intel.analytics.bigdl.nn.{Graph, Module, StaticGraph}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

class LoadBlasToDnnSpec extends BigDLSpecHelper {
  "vgg16 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    val gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    //    val blas = Vgg_16.graph(classNum, false).asInstanceOf[StaticGraph[Float]]

    val p = "/home/zhangli/workspace/zoo-model/analytics-zoo_resnet-50_imagenet_0.1.0.model"
    val modelLoad = Module.loadModule[Float](p)
    val graph = if (!modelLoad.isInstanceOf[Graph[Float]]) modelLoad.toGraph() else modelLoad
    val blas = graph.asInstanceOf[StaticGraph[Float]]
    val outBlas = blas.forward(input).toTensor[Float].clone()

    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)
    irBlas.build()

    blas.evaluate()
    irBlas.evaluate()

    val outDnn = irBlas.forward(input).toTensor[Float]

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
  }

  "inception_v1 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    RandomGenerator.RNG.setSeed(1000)
    val blas = Inception_v1_NoAuxClassifier.graph(classNum, false).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)

    val outBlas = blas.forward(input).toTensor[Float]

    gradOutput.resizeAs(outBlas).rand()

    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    RandomGenerator.RNG.setSeed(1000)
    irBlas.build()
    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-5) should be(true)
  }

  "alexnet blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    Random.setSeed(1)
    val batchSize = 4
    val input = Tensor[Float](batchSize, 3, 256, 256).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    val blas = AlexNet.graph(1000, false).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)

    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    RandomGenerator.RNG.setSeed(1000)
    irBlas.build()
    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-5) should be(true)
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
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)

    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    irBlas.build()
    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-6) should be(true)

  }

  "resnet50 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    val blas = ResNet.graph(classNum,
      T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> false, "dataset" -> DatasetType.ImageNet)).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)

    irBlas.build()
    val outBlas = blas.forward(input).toTensor[Float]
    val outDnn = irBlas.forward(input).toTensor[Float]


    gradOutput.resizeAs(outBlas).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-4) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-4) should be(true)
  }
}
