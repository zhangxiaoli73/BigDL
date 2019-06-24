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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{Linear => _, SpatialBatchNormalization => _, SpatialConvolution => _, _}
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.reflect.ClassTag

object RestNetCaffe {
  var iChannels = 0

  def graph(classNum: Int, opt: Table = T(), blasPool: Boolean = false): Graph[Float] = {
    def modelInit(graph: Graph[Float]): Unit = {
      graph.getSortedForwardExecutions.foreach(n => {
        n.element match {
          case conv
            if (conv.isInstanceOf[nn.SpatialConvolution[Float]]) =>

            // RandomGenerator.RNG.setSeed(1)

            val curModel = conv.asInstanceOf[SpatialConvolution[Float]]
            val n: Float = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
            curModel.weight.apply1(_ => RNG.normal(0, Math.sqrt(2.0f / n)).toFloat)
            curModel.bias.apply1(_ => 0.0f)

          case bn
            if (bn.isInstanceOf[nn.SpatialBatchNormalization[Float]]) =>
            val curModel = bn.asInstanceOf[SpatialBatchNormalization[Float]]
            curModel.weight.apply1(_ => 1.0f)
            curModel.bias.apply1(_ => 0.0f)

          case linear
            if (linear.isInstanceOf[nn.Linear[Float]]) =>
            linear.asInstanceOf[Linear[Float]].bias.apply1(_ => 0.0f)

          case _ =>
        }
      })
    }

    val depth = opt.get("depth").getOrElse(50)
    val shortcutType = opt.get("shortcutType").getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse[DatasetType]("dataSet", DatasetType.ImageNet)
    val optnet = opt.get("optnet").getOrElse(false)

    def shortcut(input: ModuleNode[Float], nInputPlane: Int, nOutputPlane: Int,
                 stride: Int, name: String): ModuleNode[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        val conv = ConvolutionCaffe(nInputPlane, nOutputPlane, 1, 1, stride, stride,
          optnet = optnet).setName(s"res${name}_branch1").inputs(input)
        SbnDnn(nOutputPlane).setName(s"bn${name}_branch1").inputs(conv)
      } else if (nInputPlane != nOutputPlane) {
        throw new IllegalArgumentException(s"useConv false")
      } else {
        nn.Identity[Float]().inputs(input)
      }
    }

    def bottleneck(input: ModuleNode[Float], n: Int, stride: Int, name: String = "")
    : ModuleNode[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val conv1 = ConvolutionCaffe(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet)
        .setName(s"res${name}_branch2a").inputs(input)
      val bn1 = SbnDnn(n).setName(s"bn${name}_branch2a").inputs(conv1)
      val relu1 = nn.ReLU[Float]().setName(s"res${name}_branch2a_relu").inputs(bn1)
      val conv2 = ConvolutionCaffe(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet).setName(
        s"res${name}_branch2b").inputs(relu1)
      val bn2 = SbnDnn(n).setName(s"bn${name}_branch2b").inputs(conv2)
      val relu3 = nn.ReLU[Float]().setName(s"res${name}_branch2b_relu").inputs(bn2)
      val conv3 = ConvolutionCaffe(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(
        s"res${name}_branch2c").inputs(relu3)
      val bn3 = SbnDnn(n * 4).setInitMethod(Zeros, Zeros).setName(
        s"bn${name}_branch2c").inputs(conv3)

      val short = shortcut(input, nInputPlane, n*4, stride, name)
      val cadd = nn.CAddTable[Float]().setName(s"res$name").
        inputs(Array(bn3.asInstanceOf[ModuleNode[Float]], short))
      val relu = nn.ReLU[Float]().setName(s"res${name}_relu").inputs(cadd)
      relu
    }

    def getName(i: Int, name: String): String = {
      val name1 = i match {
        case 1 => name + "a"
        case 2 => name + "b"
        case 3 => name + "c"
        case 4 => name + "d"
        case 5 => name + "e"
        case 6 => name + "f"
      }
      return name1
    }

    def layer(input: ModuleNode[Float],
              block: (ModuleNode[Float], Int, Int, String) => ModuleNode[Float],
              features: Int,
              count: Int, stride: Int = 1, name : String): ModuleNode[Float] = {
      var in = input
      for (i <- 1 to count) {
        val res = block(in, features, if (i == 1) stride else 1, getName(i, name))
        in = res
      }
      in
    }

    if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(
        50 -> ((3, 4, 6, 3), 2048,
          bottleneck: (ModuleNode[Float], Int, Int, String) => ModuleNode[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64

      val input = nn.Input[Float]()
      val conv1 = nn.SpatialConvolution[Float](3, 64, 7, 7, 2, 2, 3, 3,
        propagateBack = true).setName("conv1").inputs(input)
      val bn1 = SbnDnn(64).setName("bn_conv1").inputs(conv1)
      val relu1 = nn.ReLU[Float]().setName("conv1_relu").inputs(bn1)

      val pool1 = if (blasPool) {
        // use blas pooling parameters
        nn.SpatialMaxPooling[Float](3, 3, 2, 2, 1, 1).setName("pool1").inputs(relu1)
      } else {
        nn.SpatialMaxPooling[Float](3, 3, 2, 2).ceil().setName("pool1").inputs(relu1)
      }

      val layer1 = layer(pool1, block, 64, loopConfig._1, name = "2")
      val layer2 = layer(layer1, block, 128, loopConfig._2, 2, name = "3")
      val layer3 = layer(layer2, block, 256, loopConfig._3, 2, name = "4")
      val layer4 = layer(layer3, block, 512, loopConfig._4, 2, name = "5")
      val pool2 = nn.SpatialAveragePooling[Float](7, 7, 1, 1).setName("pool5").inputs(layer4)
      val view = nn.View[Float](nFeatures).setNumInputDims(3).inputs(pool2)

      // RandomGenerator.RNG.setSeed(1)

      // todo: linear also has regularizer
//      val output = nn.Linear[Float](nFeatures, classNum, true, L2Regularizer(1e-4),
//        L2Regularizer(1e-4)).setInitMethod(RandomNormal(0.0, 0.01), Zeros).setName("fc1000").inputs(view)

      val output = nn.Linear[Float](nFeatures, classNum, true).setInitMethod(
        RandomNormal(0.0, 0.01), Zeros).setName("fc1000").inputs(view)
      val model = Graph(Array(input), Array(output))

      modelInit(model)
      model
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataSet}")
    }
  }

  /**
    * dataset type
    * @param typeId type id
    */
  sealed abstract class DatasetType(typeId: Int)
    extends Serializable

  /**
    *  define some dataset type
    */
  object DatasetType {
    case object CIFAR10 extends DatasetType(0)
    case object ImageNet extends DatasetType(1)
  }

  /**
    * ShortcutType
    * @param typeId type id
    */
  sealed abstract class ShortcutType(typeId: Int)
    extends Serializable

  /**
    * ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
    * ShortcutType-C is used for others.
    */
  object ShortcutType{
    case object A extends ShortcutType(0)
    case object B extends ShortcutType(1)
    case object C extends ShortcutType(2)
  }
}

object ConvolutionCaffe {
  def apply(
   nInputPlane: Int,
   nOutputPlane: Int,
   kernelW: Int,
   kernelH: Int,
   strideW: Int = 1,
   strideH: Int = 1,
   padW: Int = 0,
   padH: Int = 0,
   nGroup: Int = 1,
   propagateBack: Boolean = true,
   optnet: Boolean = true,
   weightDecay: Double = 1e-4): nn.SpatialConvolution[Float] = {
    val modelType = System.getProperty("modelType", "caffeDnn")
    val wReg = if (modelType == "weightRegu") L2Regularizer[Float](weightDecay) else null
    val bReg = if (modelType == "weightRegu") L2Regularizer[Float](weightDecay) else null

    val conv = nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup, propagateBack, wReg, bReg)
    conv.setInitMethod(MsraFiller(false), Zeros)
    conv
  }
}

object SbnDnn {
  def apply(
      nOutput: Int,
      eps: Double = 1e-3,
      momentum: Double = 0.9): nn.SpatialBatchNormalization[Float] = {
    nn.SpatialBatchNormalization[Float](nOutput, eps, momentum).setInitMethod(Ones, Zeros)
  }
}