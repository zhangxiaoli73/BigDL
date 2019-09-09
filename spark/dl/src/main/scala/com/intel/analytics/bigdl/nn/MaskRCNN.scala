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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.{Convolution, Sbn}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

class MaskRCNN(val inChannels: Int,
               val outChannels: Int)(implicit ev: TensorNumeric[Float])
  extends Container[Activity, Activity, Float] {
    val anchorSizes: Array[Float] = Array[Float](32, 64, 128, 256, 512)
    val aspectRatios: Array[Float] = Array[Float](0.5f, 1.0f, 2.0f)
    val anchorStride: Array[Float] = Array[Float](4, 8, 16, 32, 64)
    val preNmsTopNTest: Int = 1000
    val postNmsTopNTest: Int = 1000
    val preNmsTopNTrain: Int = 2000
    val postNmsTopNTrain: Int = 2000
    val rpnNmsThread: Float = 0.7f
    val minSize: Int = 0
    val fpnPostNmsTopNTrain: Int = 2000
    val fpnPostNmsTopNTest: Int = 1000

    val boxResolution: Int = 7
    val maskResolution: Int = 14
    val resolution: Int = 28
    val scales: Array[Float] = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f)
    val samplingRatio: Float = 2.0f
    val boxScoreThresh: Float = 0.05f
    val boxNmsThread: Float = 0.5f
    val maxPerImage: Int = 100
    val outputSize: Int = 1024
    val numClasses: Int = 81
    val layers: Array[Int] = Array[Int](256, 256, 256, 256)
    val dilation: Int = 1
    val useGn: Boolean = false

    private val ImageInfo : Tensor[Float] = Tensor[Float](2)
    private val backbone = buildBackbone(inChannels, outChannels)
    private val rpn = RegionRroposal(inChannels, anchorSizes, aspectRatios, anchorStride,
      preNmsTopNTest, postNmsTopNTest, preNmsTopNTrain, postNmsTopNTrain, rpnNmsThread,
      minSize, fpnPostNmsTopNTest)
    private val boxHead = BoxHead(inChannels, boxResolution, scales, samplingRatio,
      boxScoreThresh, boxNmsThread, maxPerImage, outputSize, numClasses)
    private val maskHead = MaskHead(inChannels, maskResolution, scales, samplingRatio,
      layers, dilation, numClasses)

    val pool = Pooler(maskResolution, scales, samplingRatio.toInt)

    // add layer to modules
    modules.append(backbone.asInstanceOf[Module[Float]])
    modules.append(rpn.asInstanceOf[Module[Float]])
    modules.append(boxHead.asInstanceOf[Module[Float]])
    modules.append(maskHead.asInstanceOf[Module[Float]])

    def buildResNet50(): Module[Float] = {

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int,
                 useConv: Boolean = false, preName: String = ""): Module[Float] = {
      if (useConv) {
        Sequential()
          .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride).setName(preName + ".downsample.0"))
          .add(Sbn(nOutputPlane).setName(preName + ".downsample.1"))
      } else {
        Identity()
      }
    }

    def bottleneck(nInputPlane: Int, internalPlane: Int, nOutputPlane: Int,
                   stride: Int, useConv: Boolean = false, preName: String = ""): Module[Float] = {
      val s = Sequential()
        .add(Convolution(nInputPlane, internalPlane, 1, 1, stride, stride, 0, 0)
          .setName(preName + ".conv1"))
        .add(Sbn(internalPlane)
          .setName(preName + ".bn1"))
        .add(ReLU(true))
        .add(Convolution(internalPlane, internalPlane, 3, 3, 1, 1, 1, 1)
          .setName(preName + ".conv2"))
        .add(Sbn(internalPlane)
          .setName(preName + ".bn2"))
        .add(ReLU(true))
        .add(Convolution(internalPlane, nOutputPlane, 1, 1, 1, 1, 0, 0)
          .setName(preName + ".conv3"))
        .add(Sbn(nOutputPlane)
          .setName(preName + ".bn3"))

      val m = Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, nOutputPlane, stride, useConv, preName)))
        .add(CAddTable(true))
        .add(ReLU(true))
      m
    }

    def layer(count: Int, nInputPlane: Int, nOutputPlane: Int,
              downOutputPlane: Int, stride: Int = 1, preName: String): Module[Float] = {
      val s = Sequential()
          .add(bottleneck(nInputPlane, nOutputPlane, downOutputPlane, stride, true,
          preName = preName + ".0"))
      for (i <- 2 to count) {
        s.add(bottleneck(downOutputPlane, nOutputPlane, downOutputPlane, 1, false,
          preName = preName + s".${i - 1}"))
      }
      s
    }

    val model = Sequential[Float]()
      .add(Convolution(3, 64, 7, 7, 2, 2, 3, 3, optnet = false, propagateBack = false)
        .setName("conv1"))
      .add(Sbn(64).setName("bn1"))
      .add(ReLU(true))
      .add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))

    val input = Input()
    val node0 = model.inputs(input)

    val startChannels = 64
    val node1 = layer(3, startChannels, 64, inChannels, 1, "layer1").inputs(node0)
    val node2 = layer(4, inChannels, 128, inChannels * 2, 2, "layer2").inputs(node1)
    val node3 = layer(6, inChannels * 2, 256, inChannels * 4, 2, "layer3").inputs(node2)
    val node4 = layer(3, inChannels * 4, 512, inChannels * 8, 2, "layer4").inputs(node3)

    Graph(input, Array(node1, node2, node3, node4))
  }

  private def buildBackbone(inChannels: Int, outChannels: Int): Module[Float] = {
    val resnet = buildResNet50()
    val inChannelList = Array(inChannels, inChannels*2, inChannels * 4, inChannels * 8)
    val fpn = FPN(inChannelList, outChannels, topBlocks = 1)
    val model = Sequential[Float]().add(resnet).add(fpn)
    model
  }

  override def updateOutput(input: Activity): Activity = {
    val inputWidth = input.toTensor[Float].size(3)
    val inputHeight = input.toTensor[Float].size(4)
    ImageInfo.setValue(1, inputWidth)
    ImageInfo.setValue(2, inputHeight)

    val features = this.backbone.forward(input)
    val proposals = this.rpn.forward(T(features, ImageInfo))
    val boxOutput = this.boxHead.forward(T(features, proposals)).toTable
    val postProcessorBox = boxOutput[Table](2)
    val proposalsBox = postProcessorBox[Tensor[Float]](2)
    val labelsBox = postProcessorBox[Tensor[Float]](1)
//    val pp = Tensor(T(T( 359.4648,  344.7227,  419.7825,  415.5826),
//      T( 453.1522,  334.3315,  501.1705,  421.0176),
//      T( 942.7654,  292.9069,  999.7523,  358.1204),
//      T( 993.7571,  297.5816, 1018.6810,  345.1978),
//      T( 980.4796,  253.2323,  992.5759,  278.4875),
//      T( 985.8417,  252.9171,  995.6226,  281.1850),
//      T( 975.4414,  254.2575,  987.0934,  275.7443),
//      T( 896.1373,  313.2400,  931.5672,  374.0488),
//      T( 952.8575,  314.9540,  996.2098,  360.9853),
//      T( 921.4911,  284.3484,  988.0775,  343.6074),
//      T( 977.9693,  308.9249, 1004.4815,  351.7527),
//      T( 949.4941,  295.6840,  981.8075,  335.6603),
//      T( 866.8826,  337.7082,  907.4388,  386.9977),
//      T( 970.6074,  296.2614, 1015.4398,  349.3472),
//      T(1052.2827,  306.5223, 1063.3223,  337.4540),
//      T( 973.4485,  307.2058,  998.9310,  339.6187),
//      T( 871.1727,  318.4342,  923.3215,  383.5176),
//      T( 883.4898,  331.6453,  910.9511,  379.8238),
//      T( 922.5219,  285.9380,  957.7973,  338.0446),
//      T( 916.7671,  314.6682,  996.5461,  384.1572),
//      T( 989.5474,  255.2382,  999.1593,  282.3305),
//      T( 959.0654,  297.3557,  997.5435,  337.0765),
//      T( 931.9264,  344.0161, 1001.4952,  384.5714),
//      T( 944.2170,  249.2393, 1062.9771,  337.2081),
//      T( 921.8297,  249.6249, 1065.0000,  333.6861),
//      T( 587.4909,  207.1821,  744.0966,  258.9776),
//      T( 146.7835,  237.2273,  204.3738,  257.4573),
//      T( 242.7008,  240.0408,  256.0341,  253.2762),
//      T( 293.7153,  239.6023,  341.6818,  251.3944),
//      T( 493.9839,  227.9635,  517.6334,  251.1486),
//      T( 518.7712,  212.1742,  592.9564,  250.9497),
//      T( 468.7414,  230.5153,  513.8766,  248.7726),
//      T( 543.5923,  221.0722,  587.7577,  249.7338),
//      T( 381.9234,  229.3435,  451.7597,  246.1363),
//      T( 201.3457,  240.2068,  213.4665,  253.8796),
//      T( 146.5625,  247.9995,  210.5252,  258.7644),
//      T( 254.5318,  244.3242,  293.3817,  252.6587),
//      T( 471.3326,  238.6808,  514.0198,  252.7766),
//      T( 509.0908,  224.4982,  529.2344,  255.8907),
//      T( 287.4272,  242.2798,  305.3281,  251.1286),
//      T( 536.7642,  215.5787,  590.4665,  239.3698),
//      T( 125.2171,  243.9744,  198.4265,  258.3085),
//      T( 933.4740,  275.6440, 1045.8156,  304.1557),
//      T( 511.4533,  242.5052,  530.3216,  255.7836),
//      T( 126.7071,  249.5205,  149.3576,  257.5415),
//      T( 471.4401,  238.8418,  501.8425,  247.6842),
//      T( 509.2099,  234.7784,  574.7255,  258.6571),
//      T( 821.9435,  233.9751,  866.3505,  241.7147),
//      T( 212.9085,  239.7381,  253.9352,  254.2726),
//      T( 925.5219,  274.2933, 1030.0034,  327.5317),
//      T( 964.6575,  248.6210, 1063.6282,  308.8153),
//      T( 486.3723,  221.3730,  581.9128,  256.2294),
//      T( 471.9031,  228.1445,  543.9926,  254.7226),
//      T( 117.7000,  229.4729,  206.8829,  257.2673),
//      T( 348.3199,  225.7574,  462.2884,  248.5971),
//      T( 501.0192,  191.5446,  592.9208,  257.1713),
//      T( 430.5654,  382.5406,  543.2626,  412.4271),
//      T( 276.3342,  432.4406,  327.4404,  494.2536),
//      T( 264.3383,  464.8788,  283.9221,  474.4892),
//      T( 126.8237,  607.1029,  332.8095,  714.4706),
//      T( 408.4129,  400.1710,  439.2040,  454.2181),
//      T( 919.0250,  346.2679, 1002.2537,  386.6497),
//      T( 867.3154,  337.0943,  909.0655,  386.7977),
//      T( 409.4522,  400.8844,  438.9731,  456.2548),
//      T( 415.0092,  392.9026,  464.0328,  449.2130),
//      T( 127.4222,  608.2464,  332.5010,  714.0272),
//      T( 868.3989,  340.8422,  913.6042,  387.3757),
//      T( 869.8809,  344.9059,  972.6536,  387.6791),
//      T( 923.9782,  346.9974, 1002.3323,  386.0760),
//      T( 922.6125,  351.4860, 1001.9156,  385.4829),
//      T( 257.7678,  407.8460,  278.5858,  426.5238),
//      T( 924.9070,  347.3515, 1001.5031,  386.1269),
//      T( 867.5575,  344.1192,  905.8793,  387.5363)))

    val out1 = pool.forward(T(features, proposalsBox))
    val mask = this.maskHead.forward(T(features, proposalsBox, labelsBox))
    output = T(proposalsBox, labelsBox, mask)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    throw new UnsupportedOperationException("MaskRCNN model only support inference now")
  }
}
