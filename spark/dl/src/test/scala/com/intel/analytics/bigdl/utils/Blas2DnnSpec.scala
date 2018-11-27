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

import scala.util.Random

class Blas2DnnSpec extends BigDLSpecHelper {
  "vgg16 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    RandomGenerator.RNG.setSeed(1)
    val blas = Vgg_16.graph(classNum, false).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nchw)
    irBlas.build()

    RandomGenerator.RNG.setSeed(1)
    val dnn = Vgg_16.dnngraph(classNum, false).asInstanceOf[StaticGraph[Float]]
    irBlas.graph = dnn

    var gradWeight1 = blas.getParameters()._2
    var gradWeight2 = irBlas.graph.getParameters()._2

    var weight1 = blas.getParameters()._1
    var weight2 = irBlas.graph.getParameters()._1

    Equivalent.nearequals(weight1, weight2) should be (true)
    Equivalent.nearequals(gradWeight1, gradWeight2) should be (true)

//    for (i <- 0 to 2) {
//      val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
//        RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
//      var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
//        RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)
//      val outBlas = irBlas.forward(input).toTensor[Float]
//      gradOutput.resizeAs(outBlas).apply1(_ =>
//        RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)
//      blas.forward(input).toTensor[Float]
//      blas.backward(input, gradOutput).toTensor[Float]
//      irBlas.backward(input, gradOutput).toTensor[Float]
//    }

    val outBlas = blas.forward(input).toTensor[Float]
    gradOutput.resizeAs(outBlas).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]
    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    gradWeight1 = blas.getParameters()._2
    gradWeight2 = irBlas.graph.getParameters()._2

    weight1 = blas.getParameters()._1
    weight2 = irBlas.graph.getParameters()._1

    Equivalent.nearequals(weight1, weight2) should be (true)
    Equivalent.nearequals(gradWeight1, gradWeight2) should be (true)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-5) should be(true)
  }

  "Inception_Layer_v1 blas to dnn 1111" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 192, 28, 28)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    RandomGenerator.RNG.setSeed(1000)
    val in = nn.Input()
    val outNode = Inception_Layer_v1(in, 192,
      T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/")

    val blas = new StaticGraph(Seq(in), Seq(outNode))
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nchw)

    val outBlas = blas.forward(input).toTensor[Float]

    gradOutput.resizeAs(outBlas).rand()

    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    RandomGenerator.RNG.setSeed(1000)
    irBlas.build()
    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = gradInputDnn // Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-5) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-5) should be(true)

    val gradWeight1 = blas.getParameters()._2
    val gradWeight2 = irBlas.getParameters()._2

    val weight1 = blas.getParameters()._1
    val weight2 = irBlas.getParameters()._1

    Equivalent.nearequals(weight1, weight2) should be (true)
    Equivalent.nearequals(gradWeight1, gradWeight2) should be (true)
  }

  "inception_v1 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(0.0, 1.0).toFloat)

    RandomGenerator.RNG.setSeed(1000)
    val blas = Inception_v1_NoAuxClassifier.graph(classNum, false).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)

    RandomGenerator.RNG.setSeed(1000)
    irBlas.build()
    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]
    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-3) should be(true)
  }

  "alexnet blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    Random.setSeed(1)
    val batchSize = 4
    val input = Tensor[Float](batchSize, 3, 256, 256).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    val blas = AlexNet.graph(1000, false).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)

    RandomGenerator.RNG.setSeed(1000)
    irBlas.build()
     for (i <- 0 to 2) {
       val input = Tensor[Float](batchSize, 3, 256, 256).apply1(e => Random.nextFloat())
       val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())
       irBlas.forward(input).toTensor[Float]
       blas.forward(input).toTensor[Float]
       blas.backward(input, gradOutput).toTensor[Float]
       irBlas.backward(input, gradOutput).toTensor[Float]
     }
    val outDnn = irBlas.forward(input).toTensor[Float]
    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]

    //    val p1 = blas.getParametersTable()
//    val p2 = irBlas.graph.getParametersTable()
//    val keys = p1.keySet
//    for (i <- keys) {
//      val k = i.asInstanceOf[String]
//      println(k)
//      val t1 = p1[Table](k)
//      val t2 = p2[Table](k)
//      if (t1[Tensor[Float]]("weight").dim() == t2[Tensor[Float]]("weight").dim()) {
//        t1 should be(t2)
//      }
//
//    }
//
//    val weight1 = blas.getParameters()._1
//    val weight2 = irBlas.getParameters()._1
//
//    Equivalent.nearequals(weight1, weight2) should be (true)


    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-3) should be(true)

//    val gradWeight1 = blas.getParameters()._2
//    val gradWeight2 = irBlas.getParameters()._2
//
//    Equivalent.nearequals(gradWeight1, gradWeight2) should be (true)
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

    val gradWeight1 = blas.getParameters()._2
    val gradWeight2 = irBlas.getParameters()._2

    val weight1 = blas.getParameters()._1
    val weight2 = irBlas.getParameters()._1

    Equivalent.nearequals(weight1, weight2) should be (true)
    Equivalent.nearequals(gradWeight1, gradWeight2) should be (true)

  }

  "resnet50 blas to dnn" should "work properly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(-1.0, 1.0).toFloat)
    var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    val blas = ResNet.graph(classNum,
      T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> false, "dataset" -> DatasetType.ImageNet)).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = 5, outputFormats = Memory.Format.nc)

    irBlas.build()
    for (i <- 0 to 2) {
      val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
        RandomGenerator.RNG.uniform(-1.0, 1.0).toFloat)
      var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
        RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)
      irBlas.forward(input).toTensor[Float]
      blas.forward(input).toTensor[Float]
      blas.backward(input, gradOutput).toTensor[Float]
      irBlas.backward(input, gradOutput).toTensor[Float]
    }

    val outBlas = blas.forward(input).toTensor[Float]
    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputTensor, gradInputBlas, 1e-3) should be(true)
  }
}
