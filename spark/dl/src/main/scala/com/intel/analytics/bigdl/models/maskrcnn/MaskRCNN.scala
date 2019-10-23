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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.segmentation.{MaskUtils, RLEMasks}
import com.intel.analytics.bigdl.models.resnet.{Convolution, Sbn}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag
import scala.reflect.runtime._

case class MaskRCNNParams(
  anchorSizes: Array[Float] = Array[Float](32, 64, 128, 256, 512),
  aspectRatios: Array[Float] = Array[Float](0.5f, 1.0f, 2.0f),
  anchorStride: Array[Float] = Array[Float](4, 8, 16, 32, 64),
  preNmsTopNTest: Int = 1000,
  postNmsTopNTest: Int = 1000,
  preNmsTopNTrain: Int = 2000,
  postNmsTopNTrain: Int = 2000,
  rpnNmsThread: Float = 0.7f,
  minSize: Int = 0,
  boxResolution: Int = 7,
  maskResolution: Int = 14,
  scales: Array[Float] = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f),
  samplingRatio: Int = 2,
  boxScoreThresh: Float = 0.05f,
  boxNmsThread: Float = 0.5f,
  maxPerImage: Int = 100,
  outputSize: Int = 1024,
  layers: Array[Int] = Array[Int](256, 256, 256, 256),
  dilation: Int = 1,
  useGn: Boolean = false)

class MaskRCNN(val inChannels: Int,
               val outChannels: Int,
               val numClasses: Int = 81,
               val config: MaskRCNNParams = new MaskRCNNParams)(implicit ev: TensorNumeric[Float])
  extends Container[Activity, Activity, Float] {

    private val batchImgInfo : Tensor[Float] = Tensor[Float](2)
    private val backbone = buildBackbone(inChannels, outChannels)
    private val rpn = RegionProposal(inChannels, config.anchorSizes, config.aspectRatios,
      config.anchorStride, config.preNmsTopNTest, config.postNmsTopNTest, config.preNmsTopNTrain,
      config.postNmsTopNTrain, config.rpnNmsThread, config.minSize)
    private val boxHead = BoxHead(inChannels, config.boxResolution, config.scales,
      config.samplingRatio, config.boxScoreThresh, config.boxNmsThread, config.maxPerImage,
      config.outputSize, numClasses)
    private val maskHead = MaskHead(inChannels, config.maskResolution, config.scales,
      config.samplingRatio, config.layers, config.dilation, numClasses)

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

  val expectLabels = Tensor[Float](
    T( 1,  1, 27, 27, 40, 40, 41, 41, 42, 42, 42, 42, 57, 57, 57, 57, 57, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    57, 57, 59, 59, 59, 59, 59, 59, 59, 59, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 63, 63, 73, 73, 73, 73, 74, 74,
    74, 75, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76,
    76, 76))

  val expectedScores = Tensor[Float](
    T(0.9897, 0.3390, 0.0938, 0.0661, 0.0552, 0.8071, 0.3236, 0.0721, 0.1017,
    0.2883, 0.3787, 0.0858, 0.9533, 0.1806, 0.9872, 0.0926, 0.5350, 0.7239,
    0.0588, 0.8923, 0.5787, 0.1139, 0.3128, 0.0530, 0.0646, 0.1337, 0.1409,
    0.0748, 0.0653, 0.2000, 0.1046, 0.0654, 0.2355, 0.0886, 0.0651, 0.2385,
    0.1539, 0.1182, 0.8796, 0.3098, 0.0840, 0.0688, 0.0761, 0.6139, 0.1464,
    0.0906, 0.3684, 0.2710, 0.0905, 0.8570, 0.1083, 0.0691, 0.1531, 0.0829,
    0.0556, 0.7639, 0.0550, 0.1064, 0.2722, 0.2447, 0.1884, 0.2467, 0.0611,
    0.0883, 0.9855, 0.7253, 0.0580, 0.9583, 0.9121, 0.3129, 0.1780, 0.0566,
    0.0581, 0.8686, 0.8609, 0.4773, 0.3759, 0.6044, 0.0887, 0.5057, 0.1647,
    0.6098, 0.1122, 0.5854, 0.2609, 0.6136, 0.0885, 0.0601, 0.6302, 0.0511,
    0.0685, 0.0714))

  val expectedOut = Tensor[Float](
    T(T( 773.2141,  296.3130,  875.2629,  552.9583),
    T( 721.3090,  324.9946,  752.7162,  389.9704),
    T( 779.3743,  331.2612,  828.2104,  472.7323),
    T( 392.5202,  560.9691,  480.7394,  618.4543),
    T( 748.3901,  378.4600,  761.3455,  405.6181),
    T(1031.1133,  564.7753, 1100.8037,  748.8037),
    T( 589.2441,  359.1905,  606.4996,  402.1176),
    T( 676.6932,  404.4970,  699.2147,  435.6399),
    T( 659.0489,  409.6111,  676.9818,  432.5181),
    T( 313.5537,  438.9217,  346.3749,  500.0252),
    T( 676.1137,  404.8569,  699.4940,  433.0020),
    T( 747.1718,  385.9903,  762.1301,  405.0909),
    T( 683.5575,  403.5381,  808.5747,  591.3973),
    T( 406.6740,  432.7628,  555.1910,  587.7978),
    T( 554.4664,  409.7568,  658.4654,  600.3703),
    T( 569.7878,  398.3322,  640.3930,  407.5323),
    T( 558.7061,  408.4832,  645.4119,  439.1050),
    T( 592.9282,  408.7379,  636.5220,  435.5891),
    T( 655.9454,  407.5283,  701.6070,  431.7154),
    T( 768.2619,  419.3383,  831.6115,  576.7205),
    T( 677.5145,  436.7480,  725.4089,  583.5943),
    T( 688.2695,  403.3298,  802.8898,  433.7748),
    T( 812.2385,  437.6575,  834.5898,  575.3691),
    T( 597.9655,  411.4805,  758.9082,  437.1466),
    T( 623.3500,  423.3210,  698.1733,  586.9970),
    T( 629.6942,  407.1843,  653.2764,  430.6839),
    T( 554.9944,  409.3045,  571.3876,  437.7670),
    T( 771.7344,  346.0293,  890.8207,  572.5835),
    T( 559.6937,  409.6031,  589.6943,  441.4942),
    T( 778.4492,  412.4941,  823.3328,  481.5043),
    T( 689.3013,  439.5727,  723.2800,  473.1665),
    T( 606.3994,  359.7384,  884.5941,  571.9915),
    T( 549.7070,  411.6304,  610.3748,  575.2234),
    T( 781.2462,  413.5138,  868.9754,  564.1168),
    T( 716.9775,  400.7548,  796.5267,  459.0146),
    T( 600.6643,  428.5961,  662.8142,  586.5216),
    T( 663.1320,  411.0425,  711.1903,  571.7260),
    T( 716.7535,  412.2285,  777.9933,  573.2582),
    T( 432.3186,  328.5717,  500.3239,  400.2412),
    T( 625.9670,  347.8021,  654.2154,  404.8483),
    T(1085.8599,  468.2574, 1142.2051,  505.4545),
    T( 402.0193,  276.5902,  503.8932,  396.3457),
    T( 641.5566,  341.7755,  662.1937,  404.9708),
    T( 640.7853,  330.5719,  689.8660,  413.0108),
    T( 600.5721,  339.0362,  722.3686,  413.6837),
    T(1024.8645,  532.5353, 1120.7439,  742.8309),
    T( 610.1445,  431.6295,  725.0405,  440.6628),
    T( 713.0613,  402.6470,  797.7988,  413.3405),
    T( 558.3385,  404.0588,  707.2322,  434.6863),
    T( 893.2620,  659.1650, 1189.4641,  795.1557),
    T( 545.6807,  413.0492,  807.0984,  449.4756),
    T( 565.2603,  401.1346,  657.5267,  411.8421),
    T( 572.2771,  420.9645,  736.6688,  444.8611),
    T( 647.1559,  432.3820,  746.5234,  589.2875),
    T( 572.8163,  427.7688,  795.8904,  455.4704),
    T( 559.2021,  418.8543,  824.1280,  598.1257),
    T( 711.5480,  406.1088,  795.3193,  417.7105),
    T( 612.4866,  433.0777,  722.3948,  448.4608),
    T( 609.8746,  427.4308,  711.2715,  590.2625),
    T( 594.6610,  434.1769,  734.1844,  469.4973),
    T( 150.7211,  661.1986, 1101.2495,  790.3402),
    T( 865.2886,  532.9861, 1192.7415,  791.5900),
    T( 587.4130,  415.6507,  732.3098,  540.1763),
    T( 677.3251,  408.9467,  847.2313,  587.6925),
    T(  14.3146,  312.8906,  287.1800,  490.9341),
    T(1044.6263,  397.6257, 1199.5004,  536.5251),
    T( 961.5424,  415.4434, 1015.2215,  525.6513),
    T( 841.4992,  316.1215,  960.4450,  537.3554),
    T( 921.7211,  322.3628,  961.6239,  537.0927),
    T( 867.0925,  353.1292,  918.9283,  537.2457),
    T( 387.3448,  388.3776,  450.3015,  399.5246),
    T( 329.6013,  511.6688,  351.8520,  521.9038),
    T(   0.0000,  299.3038, 1125.7285,  612.2000),
    T( 842.4726,  226.9476,  863.9404,  265.5812),
    T( 452.6361,  371.9380,  476.6905,  398.0472),
    T( 633.2029,  378.6836,  649.3560,  402.3392),
    T( 649.1407,  387.1259,  662.6000,  404.5518),
    T(1031.8019,  549.2963, 1102.1632,  745.5969),
    T( 655.5341,  378.9940,  673.5594,  405.9376),
    T( 660.4626,  386.1741,  674.0916,  411.5026),
    T( 747.2353,  385.4892,  761.9861,  404.3753),
    T( 312.7620,  437.5042,  347.6777,  501.1268),
    T( 661.9788,  405.4046,  681.2984,  431.1189),
    T( 660.0209,  392.6873,  675.5620,  431.6780),
    T( 659.3785,  377.8599,  676.6752,  420.1386),
    T( 592.2844,  368.5396,  605.6808,  401.5074),
    T( 626.2499,  380.5099,  638.7192,  401.6520),
    T( 626.3574,  358.3291,  647.9949,  402.8525),
    T( 675.8460,  402.8233,  698.6690,  432.2182),
    T( 663.6592,  379.8636,  676.8243,  406.0887),
    T( 646.8254,  390.9072,  663.8757,  429.9691),
    T( 654.1721,  389.4158,  695.9028,  432.6222)))

  override def updateOutput(input: Activity): Activity = {
    val inputFeatures = input.toTable[Tensor[Float]](1)
    // image info with shape (batchSize, 4)
    // contains all images info (height, width, original height, original width)
    val imageInfo = input.toTable[Tensor[Float]](2)

    batchImgInfo.setValue(1, inputFeatures.size(3))
    batchImgInfo.setValue(2, inputFeatures.size(4))

    val features = this.backbone.forward(inputFeatures)

    if (features.toTable[Tensor[Float]](1).size(1) == 1) {
      val tmp = 0
    }

    val proposals = this.rpn.forward(T(features, batchImgInfo))
    val boxOutput = this.boxHead.forward(T(features, proposals, batchImgInfo)).toTable
    val postProcessorBox = boxOutput[Table](2)
    val labelsBox = postProcessorBox[Tensor[Float]](1)
    val proposalsBox = postProcessorBox[Table](2)
//    val proposalsBox = T(
//      Tensor[Float](T(
//      T(359.4648,  344.7227,  419.7825,  415.5826),
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
//    )
    val scores = postProcessorBox[Tensor[Float]](3)
    val masks = this.maskHead.forward(T(features, proposalsBox, labelsBox)).toTable
    if (this.isTraining()) {
      output = T(proposalsBox, labelsBox, masks, scores)
    } else {
      output = postProcessorForMaskRCNN(proposalsBox, labelsBox, masks[Tensor[Float]](2),
        scores, imageInfo)
    }

    output
  }

  @transient var binaryMask : Tensor[Float] = null
  private def postProcessorForMaskRCNN(bboxes: Table, labels: Tensor[Float],
    masks: Tensor[Float], scores: Tensor[Float], imageInfo: Tensor[Float]): Table = {
    val batchSize = bboxes.length()
    val boxesInImage = new Array[Int](batchSize)
    for (i <- 0 to batchSize - 1) {
      boxesInImage(i) = bboxes[Tensor[Float]](i + 1).size(1)
    }

    if (binaryMask == null) binaryMask = Tensor[Float]()
    val output = T()
    var start = 1
    for (i <- 0 to batchSize - 1) {
      val info = imageInfo.select(1, i + 1)
      val height = info.valueAt(1).toInt // image height after scale, no padding
      val width = info.valueAt(2).toInt // image width after scale, no padding
      val originalHeight = info.valueAt(3).toInt // Original height
      val originalWidth = info.valueAt(4).toInt // Original width

      binaryMask.resize(originalHeight, originalWidth)

      val boxNumber = boxesInImage(i)
      val maskPerImg = masks.narrow(1, start, boxNumber)
      val bboxPerImg = bboxes[Tensor[Float]](i + 1)
      val classPerImg = labels.narrow(1, start, boxNumber)
      val scorePerImg = scores.narrow(1, start, boxNumber)

      require(maskPerImg.size(1) == bboxPerImg.size(1),
        s"mask number ${maskPerImg.size(1)} should be same with box number ${bboxPerImg.size(1)}")

      // bbox resize to original size
      if (height != originalHeight || width != originalWidth) {
        BboxUtil.scaleBBox(bboxPerImg,
          originalHeight.toFloat / height, originalWidth.toFloat / width)
      }
      // mask decode to original size
      val masksRLE = new Array[RLEMasks](boxNumber)
      for (j <- 0 to boxNumber - 1) {
        binaryMask.fill(0.0f)
        MaskRCNNUtils.pasteMaskInImage(maskPerImg.select(1, j + 1), bboxPerImg.select(1, j + 1),
          binaryMask = binaryMask)
        masksRLE(j) = MaskUtils.binaryToRLE(binaryMask)
      }
      start += boxNumber

      // prepare for evaluation
      val postOutput = T()
      postOutput.update(RoiLabel.MASKS, masksRLE)
      postOutput.update(RoiLabel.BBOXES, bboxPerImg)
      postOutput.update(RoiLabel.CLASSES, classPerImg)
      postOutput.update(RoiLabel.SCORES, scorePerImg)

      output(i + 1) = postOutput
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    throw new UnsupportedOperationException("MaskRCNN model only support inference now")
  }
}

object MaskRCNN extends ContainerSerializable {
  def apply(inChannels: Int, outChannels: Int, numClasses: Int = 81,
    config: MaskRCNNParams = new MaskRCNNParams)(implicit ev: TensorNumeric[Float]): MaskRCNN =
    new MaskRCNN(inChannels, outChannels, numClasses, config)

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap

    val inChannels = DataConverter
      .getAttributeValue(context, attrMap.get("inChannels")).
      asInstanceOf[Int]

    val outChannels = DataConverter
      .getAttributeValue(context, attrMap.get("outChannels"))
      .asInstanceOf[Int]

    val numClasses = DataConverter
      .getAttributeValue(context, attrMap.get("numClasses"))
      .asInstanceOf[Int]

    // get MaskRCNNParams
    val config = MaskRCNNParams(
    anchorSizes = DataConverter
      .getAttributeValue(context, attrMap.get("anchorSizes"))
      .asInstanceOf[Array[Float]],
    aspectRatios = DataConverter
      .getAttributeValue(context, attrMap.get("aspectRatios"))
      .asInstanceOf[Array[Float]],
    anchorStride = DataConverter
      .getAttributeValue(context, attrMap.get("anchorStride"))
      .asInstanceOf[Array[Float]],
    preNmsTopNTest = DataConverter
      .getAttributeValue(context, attrMap.get("preNmsTopNTest"))
      .asInstanceOf[Int],
    postNmsTopNTest = DataConverter
      .getAttributeValue(context, attrMap.get("postNmsTopNTest"))
      .asInstanceOf[Int],
    preNmsTopNTrain = DataConverter
      .getAttributeValue(context, attrMap.get("preNmsTopNTrain"))
      .asInstanceOf[Int],
    postNmsTopNTrain = DataConverter
      .getAttributeValue(context, attrMap.get("postNmsTopNTrain"))
      .asInstanceOf[Int],
    rpnNmsThread = DataConverter
      .getAttributeValue(context, attrMap.get("rpnNmsThread"))
      .asInstanceOf[Float],
    minSize = DataConverter
      .getAttributeValue(context, attrMap.get("minSize"))
      .asInstanceOf[Int],
    boxResolution = DataConverter
      .getAttributeValue(context, attrMap.get("boxResolution"))
      .asInstanceOf[Int],
    maskResolution = DataConverter
      .getAttributeValue(context, attrMap.get("maskResolution"))
      .asInstanceOf[Int],
    scales = DataConverter
      .getAttributeValue(context, attrMap.get("scales"))
      .asInstanceOf[Array[Float]],
    samplingRatio = DataConverter
      .getAttributeValue(context, attrMap.get("samplingRatio"))
      .asInstanceOf[Int],
    boxScoreThresh = DataConverter
      .getAttributeValue(context, attrMap.get("boxScoreThresh"))
      .asInstanceOf[Float],
    maxPerImage = DataConverter
      .getAttributeValue(context, attrMap.get("maxPerImage"))
      .asInstanceOf[Int],
    outputSize = DataConverter
      .getAttributeValue(context, attrMap.get("outputSize"))
      .asInstanceOf[Int],
    layers = DataConverter
      .getAttributeValue(context, attrMap.get("layers"))
      .asInstanceOf[Array[Int]],
    dilation = DataConverter
      .getAttributeValue(context, attrMap.get("dilation"))
      .asInstanceOf[Int],
    useGn = DataConverter
      .getAttributeValue(context, attrMap.get("useGn"))
      .asInstanceOf[Boolean])

    MaskRCNN(inChannels, outChannels, numClasses, config)
      .asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    maskrcnnBuilder : BigDLModule.Builder)(implicit ev: TensorNumeric[T]) : Unit = {

    val maskrcnn = context.moduleData.module.asInstanceOf[MaskRCNN]

    val inChannelsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, inChannelsBuilder, maskrcnn.inChannels,
      universe.typeOf[Int])
    maskrcnnBuilder.putAttr("inChannels", inChannelsBuilder.build)

    val outChannelsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, outChannelsBuilder, maskrcnn.outChannels,
      universe.typeOf[Int])
    maskrcnnBuilder.putAttr("outChannels", outChannelsBuilder.build)

    val numClassesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, numClassesBuilder, maskrcnn.numClasses,
      universe.typeOf[Int])
    maskrcnnBuilder.putAttr("numClasses", numClassesBuilder.build)

    // put MaskRCNNParams
    val config = maskrcnn.config

    val anchorSizesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, anchorSizesBuilder,
      config.anchorSizes, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("anchorSizes", anchorSizesBuilder.build)

    val aspectRatiosBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, aspectRatiosBuilder,
      config.aspectRatios, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("aspectRatios", aspectRatiosBuilder.build)

    val anchorStrideBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, anchorStrideBuilder,
      config.anchorStride, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("anchorStride", anchorStrideBuilder.build)

    val preNmsTopNTestBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, preNmsTopNTestBuilder,
      config.preNmsTopNTest, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("preNmsTopNTest", preNmsTopNTestBuilder.build)

    val postNmsTopNTestBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, postNmsTopNTestBuilder,
      config.postNmsTopNTest, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("postNmsTopNTest", postNmsTopNTestBuilder.build)

    val preNmsTopNTrainBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, preNmsTopNTrainBuilder,
      config.preNmsTopNTrain, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("preNmsTopNTrain", preNmsTopNTrainBuilder.build)

    val postNmsTopNTrainBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, postNmsTopNTrainBuilder,
      config.postNmsTopNTrain, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("postNmsTopNTrain", postNmsTopNTrainBuilder.build)

    val rpnNmsThreadBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, rpnNmsThreadBuilder,
      config.rpnNmsThread, universe.typeOf[Float])
    maskrcnnBuilder.putAttr("rpnNmsThread", rpnNmsThreadBuilder.build)

    val minSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, minSizeBuilder,
      config.minSize, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("minSize", minSizeBuilder.build)

    val boxResolutionBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, boxResolutionBuilder,
      config.boxResolution, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("boxResolution", boxResolutionBuilder.build)

    val maskResolutionBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maskResolutionBuilder,
      config.maskResolution, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("maskResolution", maskResolutionBuilder.build)

    val scalesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, scalesBuilder,
      config.scales, universe.typeOf[Array[Float]])
    maskrcnnBuilder.putAttr("scales", scalesBuilder.build)

    val samplingRatioBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, samplingRatioBuilder,
      config.samplingRatio, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("samplingRatio", samplingRatioBuilder.build)

    val boxScoreThreshBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, boxScoreThreshBuilder,
      config.boxScoreThresh, universe.typeOf[Float])
    maskrcnnBuilder.putAttr("boxScoreThresh", boxScoreThreshBuilder.build)

    val maxPerImageBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maxPerImageBuilder,
      config.maxPerImage, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("maxPerImage", maxPerImageBuilder.build)

    val outputSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, outputSizeBuilder,
      config.outputSize, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("outputSize", outputSizeBuilder.build)

    val layersBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, layersBuilder,
      config.layers, universe.typeOf[Array[Int]])
    maskrcnnBuilder.putAttr("layers", layersBuilder.build)

    val dilationBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, dilationBuilder,
      config.dilation, universe.typeOf[Int])
    maskrcnnBuilder.putAttr("dilation", dilationBuilder.build)

    val useGnBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, useGnBuilder,
      config.useGn, universe.typeOf[Boolean])
    maskrcnnBuilder.putAttr("useGn", useGnBuilder.build)
  }
}
