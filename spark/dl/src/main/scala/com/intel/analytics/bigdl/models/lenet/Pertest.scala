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

package com.intel.analytics.bigdl.models.lenet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

object Pertest {

  var toGraph = false
  val batchSize = 8
  def main(args: Array[String]): Unit = {
    var name = "lenet"
    if (args.length > 0) {
      name = args(0)
    }
    println("test name: " + name)
    if (args.length > 1) toGraph = true
    println("use tograph :" + toGraph)

    name match {
      case "inception" => inception()
      case "resnet" => resnet()
      case "lenet" => lenet()
      case _ => println("error name")
    }
  }

  def resnet(): Unit = {
    RNG.setSeed(1000)
    val seqModel = ModelUntils.ResNet.basicBlockSeq(16, 16, 1, "A")
    RNG.setSeed(1000)
    val input = Input()
    val output = ModelUntils.ResNet.basicBlockSeq(16, 16, 1, "A").inputs(input)
    var funcModel = if (toGraph) {
      ModelUntils.ResNet.basicBlockSeq(16, 16, 1, "A").toGraph()
    } else {
      Graph(input, output)
    }

    val inputData = Tensor(batchSize, 16, 32, 32).rand()
    val gradients = Tensor(batchSize, 16, 32, 32).rand()

    computing(inputData, gradients, seqModel, funcModel, "resnet")
  }

  def inception(): Unit = {
    RNG.setSeed(1000)
    val seqModel = ModelUntils.Inception.inceptionLayerV1Seq(
      2, T(T(4), T(96, 128), T(16, 32), T(32)))
    RNG.setSeed(1000)
    val input = Input()
    val output = ModelUntils.Inception.inceptionLayerV1Func(
      2, T(T(4), T(96, 128), T(16, 32), T(32)))(input)
    val funcModel = if (toGraph) {
      val tmp = ModelUntils.Inception.inceptionLayerV1Seq(
        2, T(T(4), T(96, 128), T(16, 32), T(32)))
      tmp.toGraph()
    } else {
      Graph(input, output)
    }

    val inputData = Tensor(batchSize, 2, 4, 4).rand()
    val gradients = Tensor(batchSize, 256, 4, 4).rand()

    computing(inputData, gradients, seqModel, funcModel, "inception")
  }

  def lenet(): Unit = {
    RNG.setSeed(1000)
    val seqModel = ModelUntils.Lenet.lenetSeq()
    RNG.setSeed(1000)
    val funcModel = if (toGraph) {
      ModelUntils.Lenet.lenetSeq().toGraph()
    } else {
      ModelUntils.Lenet.lenetFunc()
    }
    val inputData = Tensor(batchSize, 28 * 28).rand()
    val gradients = Tensor(batchSize, 10).rand()

    computing(inputData, gradients, seqModel, funcModel, "lenet")
  }

  def computing(input: Tensor[Float], gradOutput: Tensor[Float],
                model: Module[Float], graphModel: Module[Float], name: String = ""): Unit = {
    // warm up
    for (i <- 1 to 1) {
      model.forward(input).toTensor[Float]
      model.backward(input, gradOutput)
    }
    val start1 = System.nanoTime()
    var j = 0
    for(i <- 1 to 1) {
      val t1 = model.forward(input).toTensor[Float]
      val t2 = model.backward(input, gradOutput)
    }
    println(name + " Original Module takes time " + (System.nanoTime()-start1)/1e9 + "s")
    Thread.sleep(10)

    for(i <- 1 to 1) {
      graphModel.forward(input).toTensor[Float]
      graphModel.backward(input, gradOutput)
    }
    val start2 = System.nanoTime()
    for(i <- 1 to 1) {
      val t1 = graphModel.forward(input).toTensor[Float]
      val t2 = graphModel.backward(input, gradOutput)
    }
    println(name + " Graph Module takes time " + (System.nanoTime()-start2)/1e9 + "s")
  }
}

object ModelUntils {
  object Inception {
    def inceptionLayerV1Func(inputSize: Int, config: Table)(input : ModuleNode[Float])
    : ModuleNode[Float] = {
      val conv1x1 = SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName("conv1x1").inputs(input)
      val relu1x1 = ReLU(true).inputs(conv1x1)

      val conv3x3_1 = SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName("conv3x3_1").inputs(input)
      val relu3x3_1 = ReLU(true).inputs(conv3x3_1)
      val conv3x3_2 = SpatialConvolution(
        config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
        .setName("conv3x3_2").inputs(relu3x3_1)
      val relu3x3_2 = ReLU(true).inputs(conv3x3_2)

      val conv5x5_1 = SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName("conv5x5_1").inputs(input)
      val relu5x5_1 = ReLU(true).inputs(conv5x5_1)
      val conv5x5_2 = SpatialConvolution(
        config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
        .setName("conv5x5_2").inputs(relu5x5_1)
      val relu5x5_2 = ReLU(true).inputs(conv5x5_2)

      val pool = SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil()
        .setName("pool").inputs(input)
      val convPool = SpatialConvolution(inputSize, config[Table](4)(1), 1, 1, 1, 1)
        .setName("pool_conv").inputs(pool)
      val reluPool = ReLU(true).inputs(convPool)

      JoinTable(2, 4).inputs(relu1x1, relu3x3_2, relu5x5_2, reluPool)
    }
    def inceptionLayerV1Seq(inputSize: Int, config: Table) : Module[Float] = {
      val concat = Concat(2)
      val conv1 = Sequential()
      conv1.add(SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName("conv1x1"))
      conv1.add(ReLU(true))
      concat.add(conv1)
      val conv3 = Sequential()
      conv3.add(SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName("conv3x3_1"))
      conv3.add(ReLU(true))
      conv3.add(SpatialConvolution(config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
        .setName("conv3x3_2"))
      conv3.add(ReLU(true))
      concat.add(conv3)
      val conv5 = Sequential()
      conv5.add(SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName("conv5x5_1"))
      conv5.add(ReLU(true))
      conv5.add(SpatialConvolution(config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
        .setName("conv5x5_2"))
      conv5.add(ReLU(true))
      concat.add(conv5)
      val pool = Sequential()
      pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil()
        .setName("pool"))
      pool.add(SpatialConvolution(inputSize, config[Table](4)(1), 1, 1, 1, 1).setName("pool_conv"))
      pool.add(ReLU(true))
      concat.add(pool)
      concat
    }
  }
  object ResNet {
    def basicBlockFunc(nInputPlane: Int, n: Int, stride: Int, shortcutType : String)(
      input : ModuleNode[Float]) : ModuleNode[Float] = {
      val conv1 = SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 1, 1).inputs(input)
      val bn1 = SpatialBatchNormalization(n).inputs(conv1)
      val relu1 = ReLU(true).inputs(bn1)
      val conv2 = SpatialConvolution(n, n, 3, 3, 1, 1, 1, 1).inputs(relu1)
      val bn2 = SpatialBatchNormalization(n).inputs(conv2)
      val shortcut = shortcutFunc(nInputPlane, n, stride, shortcutType)(input)
      val add = CAddTable(true).inputs(bn2, shortcut)
      val output = ReLU(true).inputs(add)
      output
    }

    def basicBlockSeq(nInputPlane: Int, n: Int, stride: Int, shortcutType : String)
    : Module[Float] = {
      val s = Sequential()
      s.add(SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 1, 1))
      s.add(SpatialBatchNormalization(n))
      s.add(ReLU(true))
      s.add(SpatialConvolution(n, n, 3, 3, 1, 1, 1, 1))
      s.add(SpatialBatchNormalization(n))

      Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcutSeq(nInputPlane, n, stride, shortcutType)))
        .add(CAddTable(true))
        .add(ReLU(true))
    }

    def shortcutFunc(nInputPlane: Int, nOutputPlane: Int, stride: Int,
                     shortcutType : String)(input : ModuleNode[Float]) : ModuleNode[Float] = {
      val useConv = shortcutType == "C" || (shortcutType == "B" && nInputPlane != nOutputPlane)

      if (useConv) {
        val conv1 = SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride)
          .inputs(input)
        val bn1 = SpatialBatchNormalization(nOutputPlane).inputs(conv1)
        bn1
      } else if (nInputPlane != nOutputPlane) {
        val pool1 = SpatialAveragePooling(1, 1, stride, stride).inputs(input)
        val mul1 = MulConstant(0f).inputs(pool1)
        val concat = JoinTable(2, 3).inputs(pool1, mul1)
        concat
      } else {
        input
      }
    }
    def shortcutSeq(nInputPlane: Int, nOutputPlane: Int, stride: Int, shortcutType : String)
    : Module[Float] = {
      val useConv = shortcutType == "C" || (shortcutType == "B" && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
          .add(SpatialBatchNormalization(nOutputPlane))
      } else if (nInputPlane != nOutputPlane) {
        Sequential()
          .add(SpatialAveragePooling(1, 1, stride, stride))
          .add(Concat(2)
            .add(Identity())
            .add(MulConstant(0f)))
      } else {
        Identity()
      }
    }
  }
  object Lenet {
    def lenetFunc(): Module[Float] = {
      val input = Reshape(Array(1, 28, 28)).inputs()
      val conv1 = SpatialConvolution(1, 6, 5, 5).inputs(input)
      val tanh1 = Tanh().inputs(conv1)
      val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh1)
      val tanh2 = Tanh().inputs(pool1)
      val conv2 = SpatialConvolution(6, 12, 5, 5).inputs(tanh2)
      val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(conv2)
      val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
      val fc1 = Linear(12 * 4 * 4, 100).inputs(reshape)
      val tanh3 = Tanh().inputs(fc1)
      val fc2 = Linear(100, 10).inputs(tanh3)
      val output = LogSoftMax().inputs(fc2)
      Graph(input, output)
    }

    def lenetSeq(): Module[Float] = {
      val seqModel = Sequential().add(Reshape(Array(1, 28, 28)))
        .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
        .add(Tanh())
        .add(SpatialMaxPooling(2, 2, 2, 2))
        .add(Tanh())
        .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
        .add(SpatialMaxPooling(2, 2, 2, 2))
        .add(Reshape(Array(12 * 4 * 4)))
        .add(Linear(12 * 4 * 4, 100).setName("fc1"))
        .add(Tanh())
        .add(Linear(100, 10).setName("fc2"))
        .add(LogSoftMax())
      seqModel
    }
  }
}