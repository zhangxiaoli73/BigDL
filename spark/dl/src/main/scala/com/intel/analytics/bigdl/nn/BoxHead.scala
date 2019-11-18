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

import breeze.linalg.Axis._1
import breeze.linalg.{*, dim}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer

class BoxHead(
  val inChannels: Int,
  val resolution: Int,
  val scales: Array[Float],
  val samplingRatio: Int,
  val scoreThresh: Float,
  val nmsThresh: Float,
  val maxPerImage: Int,
  val outputSize: Int,
  val numClasses: Int
  )(implicit ev: TensorNumeric[Float])
  extends BaseModule[Float] {

  override def buildModel(): Module[Float] = {
    val featureExtractor = this.featureExtractor(
      inChannels, resolution, scales, samplingRatio, outputSize)

    val clsPredictor = this.clsPredictor(numClasses, outputSize)
    val bboxPredictor = this.bboxPredictor(numClasses, outputSize)

    val weight = Array(10.0f, 10.0f, 5.0f, 5.0f)
    val postProcessor = new BoxPostProcessor(scoreThresh, nmsThresh,
      maxPerImage, numClasses, weight = weight)

    val features = Input()
    val proposals = Input()
    val imageInfo = Input()

    val boxFeatures = featureExtractor.setName("boxFeatures").inputs(features, proposals)
    val classLogits = clsPredictor.inputs(boxFeatures)
    val boxRegression = bboxPredictor.inputs(boxFeatures)
    val result = postProcessor.setName("result").inputs(classLogits, boxRegression, proposals, imageInfo)

    Graph(Array(features, proposals, imageInfo), Array(boxFeatures, result))
  }

//  override def updateOutput(input: Activity): Activity = {
//    val features = input.toTable[Table](1)
//    val proposals = input.toTable[Table](2)
//    val imageInfo = input.toTable[Tensor[Float]](3)
//
//
//    val boxFeatures = model.apply("boxFeatures").get
//    val classLogits = model.apply("roi_heads.box_predictor.cls_score").get
//    val boxRegression = model.apply("roi_heads.box_predictor.bbox_pred").get
//    val result = model.apply("result").get
//
//    val out1 = boxFeatures.forward(T(features, proposals))
//    val out2 = classLogits.forward(out1)
//    val out3 = boxRegression.forward(out1)
//    val out4 = result.forward(T(out2, out3, proposals, imageInfo))
//
//    output = out4 // model.updateOutput(input)
//    output
//  }

  private[nn] def clsPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val clsScore = Linear[Float](inChannels, numClass)
      .setName("roi_heads.box_predictor.cls_score")
    clsScore.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
    clsScore.bias.fill(0.0f)
    clsScore.asInstanceOf[Module[Float]]
  }

  private[nn] def bboxPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val bboxRegression = Linear[Float](inChannels, (numClass - 1) * 4)
      .setName("roi_heads.box_predictor.bbox_pred")
    bboxRegression.weight.apply1(_ => RNG.normal(0, 0.001).toFloat)
    bboxRegression.bias.fill(0.0f)
    bboxRegression.asInstanceOf[Module[Float]]
  }

  private[nn] def featureExtractor(inChannels: Int,
                                   resolution: Int,
                                   scales: Array[Float], samplingRatio: Int,
                                   representationSize: Int): Module[Float] = {
    val pooler = new Pooler(resolution, scales, samplingRatio)
    val inputSize = inChannels * math.pow(resolution, 2).toInt

    val fc1 = Linear[Float](inputSize, representationSize, withBias = true)
      .setInitMethod(Xavier, Zeros)
      .setName("roi_heads.box_head.fc1")
    val fc2 = Linear[Float](representationSize, representationSize, withBias = true)
      .setInitMethod(Xavier, Zeros)
      .setName("roi_heads.box_head.fc2")

    val model = Sequential[Float]()
      .add(pooler)
      .add(InferReshape(Array(0, -1)))
      .add(fc1)
      .add(ReLU[Float]())
      .add(fc2)
      .add(ReLU[Float]())

    model
  }
}

private[nn] class BoxPostProcessor(
    val scoreThresh: Float,
    val nmsThresh: Float,
    val maxPerImage: Int,
    val nClasses: Int,
    val weight: Array[Float] = Array(10.0f, 10.0f, 5.0f, 5.0f)
  ) (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float] {

  private val softMax = SoftMax[Float]()
  private val nmsTool: Nms = new Nms
  private val maxBBoxPerImage = 10000
  @transient  private var boxesBuf: Tensor[Float] = null
  @transient  private var concatBoxes: Tensor[Float] = null

  private var outScoresBuf = new ArrayBuffer[Tensor[Float]]
  private var outLablesBuf = new ArrayBuffer[Tensor[Float]]


  /**
   * Returns bounding-box detection results by thresholding on scores and
   * applying non-maximum suppression (NMS).
   */
  private[nn] def filterResults(boxes: Tensor[Float], scores: Tensor[Float], numOfClasses: Int,
    outLables: Tensor[Float], outBoxes: Tensor[Float], outScores: Tensor[Float]): Int = {

    val dim = (numOfClasses - 1) * 4
    boxes.resize(Array(boxes.nElement() / dim, dim))
    scores.resize(Array(scores.nElement() / numOfClasses, numOfClasses))

    val scoresNew = scores.narrow(2, 1, 80).clone()
    val arr = scoresNew.storage().array()
    val filter_mask = Tensor[Float]().resizeAs(scoresNew)
    val arr_filter = filter_mask.storage().array()

    val clsLabel = new ArrayBuffer[Float] // for index and labels
    val clsBBox = new ArrayBuffer[Float]
    val clsScore = new ArrayBuffer[Float]

    var num = 0
    for (i <- 1 to scoresNew.size(1)) {
      val singleScore = scoresNew.select(1, i) // one dim
      val singleBbox = boxes.select(1, i).resize(80, 4) // two dims

      val scoreArr = singleScore.storage().array()
      val scoreOffset = singleScore.storageOffset() - 1
      val inds = (1 to scoresNew.size(2)).filter(
        ind => scoreArr(ind - 1 + scoreOffset) > scoreThresh).toArray

      for (j <- 1 to inds.length) {
        clsLabel.append(inds(j - 1))
        clsScore.append(singleScore.valueAt(inds(j -1)))

        val bboxArr = singleBbox.select(1, inds(j - 1)).storage().array()
        val bboxOffset = singleBbox.select(1, inds(j - 1)).storageOffset() - 1
        clsBBox.append(bboxArr(bboxOffset))
        clsBBox.append(bboxArr(bboxOffset + 1))
        clsBBox.append(bboxArr(bboxOffset + 2))
        clsBBox.append(bboxArr(bboxOffset + 3))

        num += 1
      }
    }

    val tScore = Tensor[Float](clsScore.toArray, Array(num))
    val tBbox = Tensor[Float](clsBBox.toArray, Array(num, 4))
    val tBboxBackUp = tBbox.clone()

    val maxValue = tBbox.max()

    // real bbox
    for (i <- 0 until clsLabel.length) {
      val label = clsLabel(i) -1
      tBbox.select(1, i + 1).add(label * (maxValue + 1))
    }

    val clsIndex = new Array[Int](maxBBoxPerImage)
    var keepN = nmsTool.nms(tScore, tBbox, nmsThresh, clsIndex, orderWithBBox = false)
    if (maxPerImage > 0 &&  keepN > maxPerImage) keepN = maxPerImage

    outScores.resize(keepN)
    outLables.resize(keepN)
    outBoxes.resize(keepN, 4)

    for (i <- 0 until keepN) {
      outScores.setValue(i + 1, tScore.valueAt(clsIndex(i)))
      outLables.setValue(i + 1, clsLabel(clsIndex(i) - 1))
      outBoxes.select(1, i + 1).copy(tBboxBackUp.select(1, clsIndex(i)))
    }

    keepN
  }

  /**
   * input contains:the class logits, the box_regression and
   * bounding boxes that are used as reference, one for ech image
   * @param input
   * @return labels and bbox
   */
  override def updateOutput(input: Table): Table = {
    if (isTraining()) {
      output = input
      return output
    }
    val classLogits = input[Tensor[Float]](1)
    val boxRegression = input[Tensor[Float]](2)
    val bbox = if (input(3).isInstanceOf[Tensor[Float]]) {
      T(input[Tensor[Float]](3))
    } else input[Table](3)
    val imageInfo = input[Tensor[Float]](4) // height & width

    val batchSize = bbox.length()

    val boxesInImage = new Array[Int](bbox.length())
    for (i <- 0 to boxesInImage.length - 1) {
      boxesInImage(i) = bbox[Tensor[Float]](i + 1).size(1)
    }

    if (boxesBuf == null) boxesBuf = Tensor[Float]
    boxesBuf.resizeAs(boxRegression)
    if (concatBoxes == null) concatBoxes = Tensor[Float]
    concatBoxes.resize(boxesInImage.sum, 4)
    var start = 1
    for (i <- 0 to boxesInImage.length - 1) {
      val length = boxesInImage(i)
      concatBoxes.narrow(1, start, length).copy(bbox[Tensor[Float]](i + 1))
      start += length
    }

    val classProb = softMax.forward(classLogits)
    BboxUtil.decodeWithWeight(boxRegression, concatBoxes, weight, boxesBuf)
    // clip to images
    BboxUtil.clipBoxes(boxesBuf, imageInfo.valueAt(1), imageInfo.valueAt(2))

    if (output.toTable.length() == 0) {
      output.toTable(1) = Tensor[Float]() // for labels
      output.toTable(2) = T() // for bbox, use table in case of batch
      output.toTable(3) = Tensor[Float]() // for scores
    }

    val outLabels = output.toTable[Tensor[Float]](1)
    val outBBoxs = output.toTable[Table](2)
    val outScores = output.toTable[Tensor[Float]](3)

    var totalDetections = 0
    start = 1
    outLablesBuf.clear()
    outScoresBuf.clear()

    for (i <- 0 to boxesInImage.length - 1) {
      val boxNum = boxesInImage(0)
      val proposalNarrow = boxesBuf.narrow(1, start, boxNum)
      val classProbNarrow = classProb.narrow(1, start, boxNum)
      start += boxNum

      if (outBBoxs.getOrElse(i + 1, null) == null) outBBoxs(i + 1) = Tensor[Float]()

      val tempLabels = Tensor[Float]()
      val tempScores = Tensor[Float]()

      totalDetections += filterResults(proposalNarrow, classProbNarrow, nClasses,
        outBoxes = outBBoxs[Tensor[Float]](i + 1),
        outLables = tempLabels,
        outScores = tempScores)

      outLablesBuf.append(tempLabels)
      outScoresBuf.append(tempScores)
    }

    // clear others tensors in output
    for (i <- (boxesInImage.length + 1) to outBBoxs.length()) {
      outBBoxs.remove[Tensor[Float]](i)
    }

    // resize labels and scores
    if (totalDetections > 0) {
      outLabels.resize(totalDetections)
      outScores.resize(totalDetections)

      val arroutLabels = outLabels.storage().array()
      var offsetOutLabels = outLabels.storageOffset() - 1

      val arroutScores = outScores.storage().array()
      var offsetOutScores = outScores.storageOffset() - 1

      for (i <- 0 until batchSize) {
        val arrTempLabels = outLablesBuf(i).storage().array()
        val arrTempScores = outScoresBuf(i).storage().array()

        val offsetTempLabels = outLablesBuf(i).storageOffset() - 1
        val offsetTempScores = outScoresBuf(i).storageOffset() - 1

        // system copy
        System.arraycopy( arrTempLabels, offsetTempLabels, arroutLabels,
          offsetOutLabels, arrTempLabels.length)
        System.arraycopy(arrTempScores, offsetTempScores, arroutScores,
          offsetOutScores, arrTempScores.length)

        offsetOutLabels += outBBoxs[Tensor[Float]](i + 1).size(1)
        offsetOutScores += outBBoxs[Tensor[Float]](i + 1).size(1)
      }
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = gradOutput.toTable
    gradInput
  }
}

object BoxHead {
  def apply(inChannels: Int,
  resolution: Int = 7,
  scales: Array[Float] = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f),
  samplingRatio: Int = 2,
  scoreThresh: Float = 0.05f,
  nmsThresh: Float = 0.5f,
  maxPerImage: Int = 100,
  outputSize: Int = 1024,
  numClasses: Int = 81 // coco dataset class number
  ) ( implicit ev: TensorNumeric[Float]): BoxHead =
    new BoxHead(inChannels, resolution, scales, samplingRatio,
      scoreThresh, nmsThresh, maxPerImage, outputSize, numClasses)
}
