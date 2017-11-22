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

package com.intel.analytics.bigdl.example

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{BatchNormalization, BiRecurrent, BifurcateSplitTable, ConcatTable, HardTanh, Identity, JoinTable, Linear, ParallelTable, ReLU, Sequential, SpatialConvolution, Squeeze, TimeDistributed, Transpose, _}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger
import org.apache.spark.SparkContext

import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.ClassTag

class DeepSpeech2ModelLoader[T : ClassTag](depth: Int = 1, isPaperVersion: Boolean = false)
  (implicit ev: TensorNumeric[T]) {

  /**
    * The configuration of convolution for dp2.
    */
  val nInputPlane = 1
  val nOutputPlane = 1152
  val kW = 11
  val kH = 13
  val dW = 3
  val dH = 1
  val padW = 5
  val padH = 0
  val conv = SpatialConvolution(nInputPlane, nOutputPlane,
    kW, kH, dW, dH, padW, padH)

  val nOutputDim = 2
  val outputHDim = 3
  val outputWDim = 4
  val inputSize = nOutputPlane
  val hiddenSize = nOutputPlane
  val nChar = 29

  /**
    * append BiRNN layers to the deepspeech model.
    * When isPaperVersion is set to be true, the sequential will construct a DS2 w.r.t implementation from Paper.
    * Otherwise, it will construct a Nervana's DS2 version.
    * @param inputSize
    * @param hiddenSize
    * @param curDepth
    * @return
    */
  def addBRNN(inputSize: Int, hiddenSize: Int, curDepth: Int)
  : Module[T] = {
    val layers = Sequential()
    if (curDepth == 1) {
      layers
        .add(ConcatTable()
          .add(Identity[T]())
          .add(Identity[T]()))
    } else {
      layers
        .add(BifurcateSplitTable[T](3))
    }
    layers
      .add(ParallelTable[T]()
        .add(TimeDistributed[T](Linear[T](inputSize, hiddenSize, withBias = false)))
        .add(TimeDistributed[T](Linear[T](inputSize, hiddenSize, withBias = false))))
      .add(JoinTable[T](2, 2))
      .add(TimeDistributed(BatchNormalization[T](hiddenSize * 2, eps = 0.001, momentum = 0.1, affine = false)))
      .add(BiRecurrent[T](JoinTable[T](2, 2).asInstanceOf[AbstractModule[Table, Tensor[T], T]], isSplitInput = true)
      .add(RnnCellDS[T](hiddenSize, hiddenSize, HardTanh[T](0, 20, true)))
      .setName("birnn" + depth))
    layers
  }

  val brnn = Sequential()
  var i = 1
  while (i <= depth) {
    if (i == 1) {
      brnn.add(addBRNN(inputSize, hiddenSize, i))
    } else {
      brnn.add(addBRNN(hiddenSize, hiddenSize, i))
    }
    i += 1
  }

  val brnnOutputSize = if (isPaperVersion) hiddenSize else hiddenSize * 2
  val linear1 = TimeDistributed[T](Linear[T](brnnOutputSize, hiddenSize, withBias = false))
  val linear2 = TimeDistributed[T](Linear[T](hiddenSize, nChar, withBias = false))

  /**
    * The deep speech2 model.
    *****************************************************************************************
    *
    *   Convolution -> ReLU -> BiRNN (9 layers) -> Linear -> ReLUClip (HardTanh) -> Linear
    *
    *****************************************************************************************
    */
  val model = Sequential[T]()
    .add(conv)
    .add(ReLU[T]())
    .add(Transpose(Array((nOutputDim, outputWDim), (outputHDim, outputWDim))))
    .add(Squeeze(4))
    .add(brnn)
    .add(linear1)
    .add(HardTanh[T](0, 20, true))
    .add(linear2)

  def reset(): Unit = {
    conv.weight.fill(ev.fromType[Float](0.0F))
    conv.bias.fill(ev.fromType[Float](0.0F))
  }

  def setConvWeight(weights: Array[T]): Unit = {
    val temp = Tensor[T](Storage(weights), 1, Array(1, 1152, 1, 13, 11))
    conv.weight.set(Storage[T](weights), 1, conv.weight.size())
  }

  /**
    * load in the nervana's dp2 BiRNN model parameters
    * @param weights
    */
  def setBiRNNWeight(weights: Array[Array[T]]): Unit = {
    val parameters = brnn.parameters()._1
    // six tensors per brnn layer
    val numOfParams = 6
    for (i <- 0 until depth) {
      var offset = 1
      for (j <- 0 until numOfParams) {
        val length = parameters(i * numOfParams + j).nElement()
        val size = parameters(i * numOfParams + j).size
        parameters(i * numOfParams + j).set(Storage[T](weights(i)), offset, size)
        offset += length
      }
    }
  }

  /**
    * load in the nervana's dp2 Affine model parameters
    * @param weights
    * @param num
    */
  def setLinear0Weight(weights: Array[T], num: Int): Unit = {
    if (num == 0) {
      linear1.parameters()._1(0)
        .set(Storage[T](weights), 1, Array(1152, 2304))
    } else {
      linear2.parameters()._1(0)
        .set(Storage[T](weights), 1, Array(29, 1152))
    }
  }
}

object DeepSpeech2ModelLoader {

  val logger = Logger.getLogger(getClass)

  def loadModel(path: String): Module[Float] = {
    Module.load[Float](new Path(path).toString)
  }

  def loadModel(sc: SparkContext, path: String): Module[Float] = {


    /**
      ***************************************************************************
      *   Please configure your file path here:
      *   There should be 9 txt files for birnn.
      *   e.g. "/home/ywan/Documents/data/deepspeech/layer1.txt"
      ***************************************************************************
      */

    val convPath = path + "/conv.txt"
    val birnnPath = path + "/layer"
    val linear1Path = path + "/linear0.txt"
    val linear2Path = path + "/linear1.txt"

    /**
      *********************************************************
      *    set the depth to be 9
      *    for my small test, I set it to be 66.
      *********************************************************
      */

    val depth = 9
    val convFeatureSize = 1152
    val birnnFeatureSize = 1152
    val linear1FeatureSize = 2304
    val linear2FeatureSize = 1152

    /**
      *************************************************************************
      *    Loading model weights
      *    1. conv
      *    2. birnn
      *    3. linear1
      *    4. linear2
      *
      *     val depth = 9
      *     val convFeatureSize = 1152
      *     val birnnFeatureSize = 1152
      *     val linear1FeatureSize = 2304
      *     val linear2FeatureSize = 1152
      *
      *     dp2Path: place where the dp2.model is saved.
      *     Please refer to readMe.md
      *************************************************************************
      */

    logger.info("load in conv weights ..")
    val convWeights = sc.textFile(convPath)
      .map(_.split(',').map(_.toFloat)).flatMap(t => t).collect()

    logger.info("load in birnn weights ..")
    val weightsBirnn = new Array[Array[Float]](depth)
    for (i <- 0 until depth) {
      val birnnOrigin = sc.textFile(birnnPath + i + ".txt")
        .map(_.split(",").map(_.toFloat)).flatMap(t => t).collect()
      weightsBirnn(i) = convertBiRNN(birnnOrigin, birnnFeatureSize)
    }

    logger.info("load in linear1 weights ..")
    val linearOrigin0 = sc.textFile(linear1Path)
      .map(_.split(",").map(_.toFloat)).flatMap(t => t).collect()
    val weightsLinear0 = convertLinear(linearOrigin0, linear1FeatureSize)

    logger.info("load in linear2 weights ..")
    val linearOrigin1 = sc.textFile(linear2Path)
      .map(_.split(",").map(_.toFloat)).flatMap(t => t).collect()
    val weightsLinear1 = convertLinear(linearOrigin1, linear2FeatureSize)

    /**
      **************************************************************************
      *  set all the weights to the model and run the model
      *  dp2.evaluate()
      **************************************************************************
      */
    val dp2 = new DeepSpeech2ModelLoader[Float](depth)
    dp2.reset()
    dp2.setConvWeight(convert(convWeights, convFeatureSize))
    dp2.setBiRNNWeight(weightsBirnn)
    dp2.setLinear0Weight(weightsLinear0, 0)
    dp2.setLinear0Weight(weightsLinear1, 1)
    // dp2.model.save("model/dp2.bigdl")
    // println("dps2 model " + dp2.model)
    dp2.model
  }

  def convert(origin: Array[Float], channelSize: Int): Array[Float] = {
    val channel = channelSize
    val buffer = new ArrayBuffer[Float]()
    val groups = origin.grouped(channelSize).toArray

    for(i <- 0 until channel)
      for (j <- 0 until groups.length)
        buffer += groups(j)(i)
    buffer.toArray
  }

  def convertLinear(origin: Array[Float], channelSize: Int): Array[Float] = {
    val channel = channelSize
    val buffer = new ArrayBuffer[Float]()
    val groups = origin.grouped(channelSize).toArray

    for (j <- 0 until groups.length)
      for(i <- 0 until channel)
        buffer += groups(j)(i)
    buffer.toArray
  }

  def convertBiRNN(origin: Array[Float], channelSize: Int): Array[Float] = {
    val nIn = channelSize
    val nOut = channelSize
    val heights = 2 * (nIn + nOut + 1)
    val widths = nOut

    val buffer = new ArrayBuffer[Float]()
    val groups = origin.grouped(nOut).toArray

    /**
      * left-to-right i2h
      * right-to-left i2h
      *
      * left-to-right h2h
      * left-to-right bias
      *
      * right-to-left h2h
      * right-to-left bias
      */
    for (i <- 0 until 2 * nIn + nOut) {
      for (j <- 0 until nOut) {
        buffer += groups(i)(j)
      }
    }

    for (i <- 2 * (nIn + nOut + 1) - 2 until 2 * (nIn + nOut + 1) - 1) {
      for (j <- 0 until nOut) {
        buffer += groups(i)(j)
      }
    }

    for (i <- (2 * nIn + nOut) until (2 * nIn + 2 * nOut)) {
      for (j <- 0 until nOut) {
        buffer += groups(i)(j)
      }
    }

    for (i <- 2 * (nIn + nOut + 1) - 1 until 2 * (nIn + nOut + 1)) {
      for (j <- 0 until nOut) {
        buffer += groups(i)(j)
      }
    }
    buffer.toArray
  }
}
