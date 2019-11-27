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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.RoiImageInfo
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
  samplingRatio: Int = 0,
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
  initModules()

  // add layer to modules
  private def initModules(): Unit = {
      val backbone = buildBackbone(inChannels, outChannels)
      val rpn = RegionProposal(inChannels, config.anchorSizes, config.aspectRatios,
        config.anchorStride, config.preNmsTopNTest, config.postNmsTopNTest, config.preNmsTopNTrain,
        config.postNmsTopNTrain, config.rpnNmsThread, config.minSize)
      val boxHead = BoxHead(inChannels, config.boxResolution, config.scales,
        config.samplingRatio, config.boxScoreThresh, config.boxNmsThread, config.maxPerImage,
        config.outputSize, numClasses)
      val maskHead = MaskHead(inChannels, config.maskResolution, config.scales,
        config.samplingRatio, config.layers, config.dilation, numClasses)

      modules.append(backbone.asInstanceOf[Module[Float]])
      modules.append(rpn.asInstanceOf[Module[Float]])
      modules.append(boxHead.asInstanceOf[Module[Float]])
      modules.append(maskHead.asInstanceOf[Module[Float]])
    }

  private def buildResNet50(): Module[Float] = {

    def convolution (nInputPlane: Int, nOutputPlane: Int, kernelW: Int, kernelH: Int,
      strideW: Int = 1, strideH: Int = 1, padW: Int = 0, padH: Int = 0,
      nGroup: Int = 1, propagateBack: Boolean = true): SpatialConvolution[Float] = {
        val conv = SpatialConvolution[Float](nInputPlane, nOutputPlane, kernelW, kernelH,
          strideW, strideH, padW, padH, nGroup, propagateBack, withBias = false)
        conv.setInitMethod(MsraFiller(false), Zeros)
        conv
      }

    def sbn(nOutput: Int, eps: Double = 1e-5, momentum: Double = 0.1, affine: Boolean = true)
      : SpatialBatchNormalization[Float] = {
        SpatialBatchNormalization[Float](nOutput, eps, momentum, affine).setInitMethod(Ones, Zeros)
      }

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int,
                 useConv: Boolean = false, preName: String = ""): Module[Float] = {
      if (useConv) {
        Sequential()
          .add(convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride).setName(preName + ".shortcut"))
          .add(sbn(nOutputPlane).setName(preName + ".shortcut.norm"))
      } else {
        Identity()
      }
    }

    def bottleneck(nInputPlane: Int, internalPlane: Int, nOutputPlane: Int,
                   stride: Int, useConv: Boolean = false, preName: String = ""): Module[Float] = {
      val s = Sequential()
        .add(convolution(nInputPlane, internalPlane, 1, 1, stride, stride, 0, 0)
          .setName(preName + ".conv1"))
        .add(sbn(internalPlane)
          .setName(preName + ".conv1.norm"))
        .add(ReLU(true))
        .add(convolution(internalPlane, internalPlane, 3, 3, 1, 1, 1, 1)
          .setName(preName + ".conv2"))
        .add(sbn(internalPlane)
          .setName(preName + ".conv2.norm"))
        .add(ReLU(true))
        .add(convolution(internalPlane, nOutputPlane, 1, 1, 1, 1, 0, 0)
          .setName(preName + ".conv3"))
        .add(sbn(nOutputPlane)
          .setName(preName + ".conv3.norm"))

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
      .add(convolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = false)
        .setName("backbone.bottom_up.stem.conv1"))
      .add(sbn(64).setName("backbone.bottom_up.stem.conv1.norm"))
      .add(ReLU(true))
      .add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))

    val input = Input()
    val node0 = model.inputs(input)

    val startChannels = 64
    val node1 = layer(3, startChannels, 64, inChannels, 1, "backbone.bottom_up.res2").inputs(node0)
    val node2 = layer(4, inChannels, 128, inChannels * 2, 2, "backbone.bottom_up.res3").inputs(node1)
    val node3 = layer(6, inChannels * 2, 256, inChannels * 4, 2, "backbone.bottom_up.res4").inputs(node2)
    val node4 = layer(3, inChannels * 4, 512, inChannels * 8, 2, "backbone.bottom_up.res5").inputs(node3)

    Graph(input, Array(node1, node2, node3, node4))
  }

  private def buildBackbone(inChannels: Int, outChannels: Int): Module[Float] = {
    val resnet = buildResNet50()
    val inChannelList = Array(inChannels, inChannels*2, inChannels * 4, inChannels * 8)
    val fpn = FPN(inChannelList, outChannels, topBlocks = 1)
    val model = Sequential[Float]().add(resnet).add(fpn)
    model
  }

  def debugTests(): Table = {
    val postOutput = T()
    postOutput.update(RoiImageInfo.MASKS, GroudTrue.getMasks())
    postOutput.update(RoiImageInfo.BBOXES, GroudTrue.bbox)
    postOutput.update(RoiImageInfo.CLASSES, GroudTrue.labels)
    postOutput.update(RoiImageInfo.SCORES, GroudTrue.scores)

    T(postOutput)
  }

  val expected = Tensor[Float]()
  override def updateOutput(input: Activity): Activity = {
//    output = debugTests
//    return output

    val inputFeatures = input.toTable[Tensor[Float]](1)
    // image info with shape (batchSize, 4)
    // contains all images info (height, width, original height, original width)
    val imageInfo = input.toTable[Tensor[Float]](2)

    println(s"batchSize11 ${inputFeatures.size(1)}")
    // get each layer from modules
    val backbone = modules(0)
    val rpn = modules(1)
    val boxHead = modules(2)
    val maskHead = modules(3)

    batchImgInfo.setValue(1, inputFeatures.size(3))
    batchImgInfo.setValue(2, inputFeatures.size(4))

    val features = backbone.forward(inputFeatures)
    val proposals = rpn.forward(T(features, batchImgInfo))
    val boxOutput = boxHead.forward(T(features, proposals, batchImgInfo)).toTable
    val postProcessorBox = boxOutput[Table](2)
    val labelsBox = postProcessorBox[Tensor[Float]](1)
    val proposalsBox = postProcessorBox[Table](2)
    val scores = postProcessorBox[Tensor[Float]](3)

//    val proposalsBox = T(Tensor[Float](T(
//      T(2.7671e+02, 1.8681e+02, 4.1434e+02, 3.9764e+02),
//      T(3.9497e+02, 2.6825e+02, 4.8354e+02, 3.1071e+02),
//      T(5.9488e+02, 2.7918e+01, 6.3970e+02, 2.5170e+02),
//      T(3.0060e+02, 8.7962e+01, 3.9379e+02, 2.2026e+02),
//      T(4.9795e+02, 0.0000e+00, 5.5702e+02, 1.4416e+01),
//      T(4.1971e+02, 1.2460e-01, 4.6864e+02, 1.3768e+01),
//      T(1.8132e+02, 0.0000e+00, 2.5347e+02, 1.1476e+01),
//      T(2.6109e+02, 0.0000e+00, 3.3091e+02, 1.2007e+01),
//      T(5.4183e+02, 0.0000e+00, 6.0753e+02, 1.5087e+01),
//      T(3.5663e+02, 0.0000e+00, 3.9615e+02, 1.2099e+01),
//      T(1.1122e+02, 0.0000e+00, 1.7353e+02, 1.0241e+01),
//      T(5.2678e+01, 2.9813e-01, 1.2661e+02, 9.1792e+00),
//      T(1.2347e+01, 0.0000e+00, 5.5106e+01, 7.6769e+00),
//      T(2.9343e+02, 1.8680e+02, 3.2595e+02, 2.3129e+02),
//      T(5.7435e+02, 2.2459e-01, 6.1081e+02, 1.5237e+01),
//      T(4.5480e+02, 0.0000e+00, 5.0539e+02, 1.3854e+01),
//      T(5.2677e+02, 4.1409e-01, 5.7820e+02, 1.5827e+01),
//      T(6.2021e+02, 2.4268e-01, 6.4000e+02, 1.4627e+01),
//      T(1.0472e+02, 2.2304e-01, 1.4137e+02, 9.2909e+00),
//      T(1.3045e+02, 1.2613e-01, 1.7103e+02, 7.7234e+00),
//      T(5.9362e+02, 0.0000e+00, 6.3642e+02, 1.6144e+01),
//      T(1.1240e+00, 0.0000e+00, 1.4177e+01, 7.2199e+00),
//      T(2.4598e+02, 2.0285e+02, 2.8386e+02, 2.3946e+02),
//      T(4.0319e+02, 7.8811e-02, 5.0200e+02, 1.4211e+01),
//      T(9.3303e-01, 0.0000e+00, 3.7506e+01, 7.4055e+00),
//      T(1.2915e+02, 1.2419e+00, 1.7278e+02, 1.0354e+01),
//      T(3.7341e+02, 1.8630e+00, 3.9634e+02, 1.2259e+01),
//      T(3.4623e+02, 0.0000e+00, 3.7484e+02, 1.0556e+01),
//      T(2.9265e+02, 0.0000e+00, 3.3517e+02, 1.0797e+01),
//      T(5.3991e+02, 6.7314e+00, 5.7600e+02, 1.5891e+01),
//      T(3.4741e+02, 7.2838e-02, 4.7991e+02, 1.3621e+01),
//      T(5.5931e+01, 1.2811e+00, 1.0946e+02, 9.4318e+00),
//      T(7.8640e+00, 0.0000e+00, 9.3605e+01, 8.5584e+00),
//      T(5.9713e+02, 6.2347e+01, 6.3830e+02, 1.2057e+02),
//      T(6.0933e+02, 6.4455e+01, 6.3941e+02, 1.0485e+02),
//      T(3.3672e+01, 3.6494e-01, 1.8292e+02, 1.0487e+01))))

//    val proposalsBox = T(
//      Tensor[Float](T(T(5.1839e+02, 3.4999e+02, 7.7625e+02, 7.4500e+02),
//      T(7.3996e+02, 5.0258e+02, 9.0587e+02, 5.8212e+02),
//      T(1.1145e+03, 5.2305e+01, 1.1984e+03, 4.7157e+02),
//      T(5.6315e+02, 1.6480e+02, 7.3774e+02, 4.1266e+02),
//      T(9.3288e+02, 0.0000e+00, 1.0435e+03, 2.7010e+01),
//      T(7.8629e+02, 2.3343e-01, 8.7798e+02, 2.5796e+01),
//      T(3.3969e+02, 0.0000e+00, 4.7486e+02, 2.1501e+01),
//      T(4.8914e+02, 0.0000e+00, 6.1993e+02, 2.2495e+01),
//      T(1.0151e+03, 0.0000e+00, 1.1382e+03, 2.8266e+01),
//      T(6.6812e+02, 0.0000e+00, 7.4217e+02, 2.2667e+01),
//      T(2.0836e+02, 0.0000e+00, 3.2510e+02, 1.9186e+01),
//      T(9.8689e+01, 5.5856e-01, 2.3720e+02, 1.7198e+01),
//      T(2.3131e+01, 0.0000e+00, 1.0324e+02, 1.4383e+01),
//      T(5.4972e+02, 3.4998e+02, 6.1065e+02, 4.3333e+02),
//      T(1.0760e+03, 4.2077e-01, 1.1443e+03, 2.8547e+01),
//      T(8.5204e+02, 0.0000e+00, 9.4681e+02, 2.5957e+01),
//      T(9.8687e+02, 7.7581e-01, 1.0832e+03, 2.9652e+01),
//      T(1.1619e+03, 4.5467e-01, 1.1990e+03, 2.7405e+01),
//      T(1.9618e+02, 4.1787e-01, 2.6485e+02, 1.7407e+01),
//      T(2.4438e+02, 2.3631e-01, 3.2042e+02, 1.4470e+01),
//      T(1.1121e+03, 0.0000e+00, 1.1923e+03, 3.0247e+01),
//      T(2.1058e+00, 0.0000e+00, 2.6560e+01, 1.3527e+01),
//      T(4.6084e+02, 3.8005e+02, 5.3180e+02, 4.4863e+02),
//      T(7.5535e+02, 1.4765e-01, 9.4047e+02, 2.6624e+01),
//      T(1.7480e+00, 0.0000e+00, 7.0265e+01, 1.3874e+01),
//      T(2.4195e+02, 2.3268e+00, 3.2368e+02, 1.9398e+01),
//      T(6.9956e+02, 3.4903e+00, 7.4251e+02, 2.2968e+01),
//      T(6.4863e+02, 0.0000e+00, 7.0225e+02, 1.9777e+01),
//      T(5.4827e+02, 0.0000e+00, 6.2792e+02, 2.0229e+01),
//      T(1.0115e+03, 1.2612e+01, 1.0791e+03, 2.9772e+01),
//      T(6.5084e+02, 1.3646e-01, 8.9908e+02, 2.5519e+01),
//      T(1.0478e+02, 2.4001e+00, 2.0507e+02, 1.7671e+01),
//      T(1.4733e+01, 0.0000e+00, 1.7536e+02, 1.6034e+01),
//      T(1.1187e+03, 1.1681e+02, 1.1958e+03, 2.2589e+02),
//      T(1.1415e+03, 1.2076e+02, 1.1979e+03, 1.9645e+02),
//      T(6.3083e+01, 6.8372e-01, 3.4269e+02, 1.9648e+01))))
//
//    val scores = Tensor[Float](
//      T(0.9990, 0.9977, 0.9969, 0.9880, 0.9439, 0.9381, 0.9042, 0.8925, 0.8095,
//      0.8014, 0.7383, 0.6898, 0.5677, 0.5594, 0.3627, 0.3293, 0.2337, 0.2194,
//      0.2026, 0.1576, 0.1575, 0.1527, 0.1490, 0.1449, 0.0969, 0.0893, 0.0824,
//      0.0783, 0.0670, 0.0625, 0.0623, 0.0620, 0.0594, 0.0593, 0.0561, 0.0512))
//
//    val labelsBox = Tensor[Float](
//      T(0, 38,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,
//      0,  0,  0,  0, 32,  0,  0, 56,  0,  0,  0,  0,  0, 56,  0, 24, 24,  0)).add(1.0f)

    if (labelsBox.isEmpty || labelsBox.size(1) > 0) {
      val masks =  maskHead.forward(T(features, proposalsBox, labelsBox)).toTable
      if (this.isTraining()) {
        output = T(proposalsBox, labelsBox, masks, scores)
      } else {
        output = postProcessorForMaskRCNN(proposalsBox, labelsBox, masks[Tensor[Float]](2),
          scores, imageInfo)
      }
    } else { // detect nothing
      for (i <- 1 to inputFeatures.size(1)) {
        output.toTable(i) = T()
      }
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

      // prepare for evaluation
      val postOutput = T()

      val boxNumber = boxesInImage(i)
      if (boxNumber > 0) {
        val maskPerImg = masks.narrow(1, start, boxNumber)
        val bboxPerImg = bboxes[Tensor[Float]](i + 1)
        val classPerImg = labels.narrow(1, start, boxNumber)
        val scorePerImg = scores.narrow(1, start, boxNumber)

        require(maskPerImg.size(1) == bboxPerImg.size(1), s"mask number ${maskPerImg.size(1)} " +
          s"should be the same with box number ${bboxPerImg.size(1)}")

        // resize bbox to original size
        if (height != originalHeight || width != originalWidth) {
          BboxUtil.scaleBBox(bboxPerImg,
            originalHeight.toFloat / height, originalWidth.toFloat / width)
        }
        // decode mask to original size
        val masksRLE = new Array[RLEMasks](boxNumber)
        for (j <- 0 to boxNumber - 1) {
          binaryMask.fill(0.0f)
          Utils.decodeMaskInImage(maskPerImg.select(1, j + 1).clone(),
            bboxPerImg.select(1, j + 1), binaryMask = binaryMask)
          masksRLE(j) = MaskUtils.binaryToRLE(binaryMask)
        }
        start += boxNumber

        println(s"bbbbbbbbbbbbbbbbb ${boxNumber}")

//        val expected = GroudTrue.getMasks(height = originalHeight, wide = originalWidth)
//        // check
//        require(expected.length == masksRLE.length)
//        for (m <- 0 to expected.length - 1) {
//          val t1 = expected(m).counts
//          val t2 = masksRLE(m).counts
//          if (t1.length != t2.length) {
//            val tmp1 = 0
//          } else {
//            (t1, t2).zipped.map { case (tt1, tt2) =>
//                if (math.abs(tt1-tt2) > 2) {
//                  val tmp2 = 0
//                }
//            }
//          }
//        }
        postOutput.update(RoiImageInfo.MASKS, masksRLE)
        postOutput.update(RoiImageInfo.BBOXES, bboxPerImg)
        postOutput.update(RoiImageInfo.CLASSES, classPerImg)
        postOutput.update(RoiImageInfo.SCORES, scorePerImg)
      }

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

    val maskrcnn = MaskRCNN(inChannels, outChannels, numClasses, config)
      .asInstanceOf[Container[Activity, Activity, T]]
    maskrcnn.modules.clear()
    loadSubModules(context, maskrcnn)

    maskrcnn
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

    serializeSubModules(context, maskrcnnBuilder)
  }
}
