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

package com.intel.analytics.bigdl.nn.mkldnn.models

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.tf.TensorflowLoader

/**
 * This example show how to load a tensorflow model defined in slim and
 * use it to do prediction
 */
object LoadTest {

  def reOrder(shape: Array[Int], outShape: Array[Int]): Module[Float] = {
    RNG.setSeed(100)
    val model1 = mkldnn.Sequential()
    // model1.add(mkldnn.Input(shape, Memory.Format.nchw))
    val nchwShape = Array(shape(0), shape(3), shape(1), shape(2))
    val nchwShape2 = Array(outShape(0), outShape(3), outShape(1), outShape(2))

    model1.add(mkldnn.ReorderMemory(mkldnn.HeapData(nchwShape, Memory.Format.nhwc),
      mkldnn.HeapData(nchwShape, Memory.Format.nchw), null, null))
    model1.add(mkldnn.ReorderMemory(outputFormat = mkldnn.HeapData(outShape, Memory.Format.nhwc),
      gradInputFormat = mkldnn.HeapData(nchwShape2, Memory.Format.nchw)).setName("testRE"))

    model1
  }

  def dnn(shape: Array[Int], outShape: Array[Int]): Module[Float] = {
    RNG.setSeed(100)
    val model1 = mkldnn.Sequential()
    // model1.add(mkldnn.Input(shape, Memory.Format.nchw))
    val nchwShape = Array(shape(0), shape(3), shape(1), shape(2))
    val nchwShape2 = Array(outShape(0), outShape(3), outShape(1), outShape(2))

    model1.add(mkldnn.ReorderMemory(mkldnn.HeapData(nchwShape, Memory.Format.nhwc),
      mkldnn.HeapData(nchwShape, Memory.Format.nchw), null, null))
    // model1.add(mkldnn.Input(outShape, Memory.Format.nchw))
    model1.add(mkldnn.SpatialConvolution(3, 32, 5, 5, 1, 1, -1, -1))
    model1.add(mkldnn.ReLU())
    model1.add(mkldnn.MaxPooling(2, 2, 2, 2, 0, 0))
    model1.add(mkldnn.SpatialConvolution(32, 64, 5, 5, 1, 1, -1, -1))
    model1.add(mkldnn.ReLU())
    model1.add(mkldnn.MaxPooling(2, 2, 2, 2, 0, 0))
    model1.add(mkldnn.ReorderMemory(outputFormat = mkldnn.HeapData(outShape, Memory.Format.nhwc),
      gradInputFormat = mkldnn.HeapData(nchwShape2, Memory.Format.nchw)).setName("test"))

    model1
  }


  def dnnNCHW(shape: Array[Int], outShape: Array[Int]): Module[Float] = {
    RNG.setSeed(100)
    val model1 = mkldnn.Sequential()
    model1.add(mkldnn.Input(shape, Memory.Format.nchw))
    model1.add(mkldnn.SpatialConvolution(3, 32, 5, 5, 1, 1, -1, -1))
    model1.add(mkldnn.ReLU())
    model1.add(mkldnn.MaxPooling(2, 2, 2, 2, 0, 0))
    model1.add(mkldnn.SpatialConvolution(32, 64, 5, 5, 1, 1, -1, -1))
    model1.add(mkldnn.ReLU())
    model1.add(mkldnn.MaxPooling(2, 2, 2, 2, 0, 0))
    model1.add(mkldnn.ReorderMemory(outputFormat = mkldnn.HeapData(outShape, Memory.Format.nchw)
    ).setName("test"))

    model1
  }


  def nonDnn(): Module[Float] = {
    RNG.setSeed(100)
    import com.intel.analytics.bigdl.nn
    val model1 = nn.Sequential()
    model1.add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, -1, -1, format = DataFormat("nhwc")))
    model1.add(nn.ReLU())
    model1.add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0, format = DataFormat("nhwc")))
    model1.add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, -1, -1, format = DataFormat("nhwc")))
    model1.add(nn.ReLU())
    model1.add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0, format = DataFormat("nhwc")))

    model1
  }

  def main(args: Array[String]): Unit = {
    // require(args.length == 1, "Please input the model path as the first argument")

//    val p = "/home/zhangli/workspace/vgg/model.pb"
//    val input = Seq("input_node")
//    val output = Seq("vgg_16/fc8/squeezed")

    val p = "/home/zhangli/workspace/vgg/lenet.pb"
    val input = Seq("input_node")
    val output = Seq("LeNet/pool2/MaxPool")


    println("done")

    val inNCHW = Tensor[Float](1, 3, 32, 32).rand() // NCHW
    val inNHWC = Tensor(inNCHW.size()).copy(inNCHW).transpose(2, 4).transpose(2, 3).contiguous() // NHWC

    val modelTF = TensorflowLoader.load(p, input, output, ByteOrder.LITTLE_ENDIAN)
    modelTF.getParameters()._1.fill(0.001f)
    modelTF.getParameters()._2.fill(0.1f)

    val outTF = modelTF.forward(inNHWC)

    val modelNHWC = nonDnn()


//    model1.getParameters()._1.copy(model.getParameters()._1)
//    model1.getParameters()._2.copy(model.getParameters()._2)

    modelNHWC.getParameters()._1.fill(0.001f)
    modelNHWC.getParameters()._2.fill(0.1f)

    val outNHWC = modelNHWC.forward(inNHWC).toTensor[Float] // NHWC

    val dnnModel = dnn(inNHWC.size(), outNHWC.size())
    dnnModel.asInstanceOf[MklDnnContainer].compile(
      Phase.TrainingPhase, Array(HeapData(inNHWC.size(), Memory.Format.nhwc)))

    val dnnReorder = reOrder(inNHWC.size(), inNHWC.size())
    dnnReorder.asInstanceOf[MklDnnContainer].compile(
      Phase.TrainingPhase, Array(HeapData(inNHWC.size(), Memory.Format.nhwc)))

    val outReorder = dnnReorder.forward(inNHWC)

    val outShapeNCHW = Array(outNHWC.size(1), outNHWC.size(4), outNHWC.size(2), outNHWC.size(3))
    val dnnNCHWModel = dnnNCHW(inNCHW.size(), outShapeNCHW)
    dnnNCHWModel.asInstanceOf[MklDnnContainer].compile(
      Phase.TrainingPhase, Array(HeapData(inNCHW.size(), Memory.Format.nchw)))

    dnnModel.getParameters()._1.fill(0.001f)
    dnnModel.getParameters()._2.fill(0.1f)

    dnnNCHWModel.getParameters()._1.fill(0.001f)
    dnnNCHWModel.getParameters()._2.fill(0.1f)

    val outDnn = dnnModel.forward(inNHWC).toTensor[Float]
    val outDnnNCHW = dnnNCHWModel.forward(inNCHW).toTensor[Float]

    val outReal = outDnnNCHW.transpose(2, 3).transpose(3, 4).contiguous().clone()
    // check result
    val eps = outReal.sub(outNHWC).pow(2).sum()
    println("with transpose ********** " + eps)

    val eps1 = outDnn.sub(outNHWC).pow(2).sum()
    println("with reorder ********** " + eps1)

    println("done")

//    val result = model.forward(Tensor(1, 1, 28, 28).rand())
//    println(result)

    val tmp = 0
  }

//  def main(args: Array[String]): Unit = {
//    // require(args.length == 1, "Please input the model path as the first argument")
//
//    //    val p = "/home/zhangli/workspace/vgg/model.pb"
//    //    val input = Seq("input_node")
//    //    val output = Seq("vgg_16/fc8/squeezed")
//
//    val p = "/home/zhangli/workspace/vgg/lenet.pb"
//    val input = Seq("input_node")
//    val output = Seq("LeNet/pool2/MaxPool")
//
//    //    val model = Module.loadTF(p, Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))
//    //    val model = Module.loadTF(args(0), Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))
//    //    val result = model.forward(Tensor(1, 1, 28, 28).rand())
//    //    println(result)
//
//    val shape = Array(4, 1, 7, 7)
//    val shape2 = Array(4, 1, 5, 5)
//    val in = Tensor[Float](shape).rand() // NCHW
//    //    val in2 = Tools.toNCHW(in, HeapData(shape, Memory.Format.nhwc))
//    //    in2.resize(Array(4, 7, 1, 7))
//
//    val in2 = Tensor(in.size()).copy(in).transpose(2, 4).transpose(2, 3).contiguous() //NHWC
//
//
//    // val m1 = SpatialConvolution(3, 32, 5, 5, 1, 1, -1, -1, format = DataFormat("NCHW"))
//    RNG.setSeed(100)
//    val m1 = SpatialConvolution(1, 1, 3, 3, 1, 1, 0, 0, format = DataFormat("NHWC"))
//    import com.intel.analytics.bigdl.nn.mkldnn
//    RNG.setSeed(100)
//    val m2 = mkldnn.SpatialConvolution(1, 1, 3, 3, 1, 1, 0, 0, format = DataFormat("NCHW"))
//
//    val p1 = m1.getParameters()
//    val p2 = m2.getParameters()
//
//
//    m2.setRuntime(new MklDnnRuntime)
//    m2.initFwdPrimitives(Array(HeapData(shape, Memory.Format.nchw)), TrainingPhase)
//    m2.initBwdPrimitives(Array(HeapData(shape2, Memory.Format.nchw)), TrainingPhase)
//    m2.initGradWPrimitives(Array(HeapData(shape2, Memory.Format.nchw)), TrainingPhase)
//
//
//    val out1 = m1.forward(in2).toTensor[Float] // NHWC
//    val out2 = m2.forward(in).toTensor[Float]
//
//    val tmp1 = out1.transpose(2, 4).transpose(3, 4).contiguous() // NCHW
//    val tmp2 = Tools.toNCHW(out2, m2.outputFormats()(0)) // NCHW
//
//    val gradOutputNHWC = Tensor(out1.size()).fill(1.0f)
//    val gradOutput = Tensor(out2.size()).fill(1.0f)
//
//    val grad1 = m1.backward(in2, gradOutputNHWC).toTensor[Float] // NHWC
//    val grad2 = m2.backward(in, gradOutput).toTensor[Float]
//
//    val gradTmp1 = grad1.transpose(2, 4).transpose(3, 4).contiguous() // NCHW
//    val gradTmp2 = Tools.toNCHW(grad2, m2.gradInputFormats()(0)) // NCHW
//
//
//    println("done")
//
//    //    val model = TensorflowLoader.load(p, input, output, ByteOrder.LITTLE_ENDIAN)
//    //
//    //    val m = model.asInstanceOf[Graph[Float]].modules
//    //
//    //    val lout222 = model.forward(in)
//    //
//    //    val modelDef = TensorflowLoader.loadToIR(p, input, output)
//    //    modelDef.build()
//    //    val m2 = modelDef.graph
//    //
//    //    val p1 = model.getParametersTable()
//    //    val p2 = m2.getParametersTable()
//    //
//    ////    modelDef.graph.asInstanceOf[DnnGraph].compile(
//    ////      Phase.TrainingPhase, Array(HeapData(Array(1, 3, 32, 32),
//    ////        Memory.Format.nchw)))
//    //
//    //    val out1 = model.forward(in).toTensor[Float] // nhwc
//    //    val out2 = m2.forward(input2).toTensor[Float] // nchw
//    //
//    //    // out1 to nchw
//    //    val tt2 = Tools.toNCHW(out1, HeapData(Array(1, 8, 8, 64), Memory.Format.nhwc))
//    //    tt2.resize(Array(1, 64, 8, 8))
//    //
//    ////    val result = model.forward(Tensor(1, 1, 28, 28).rand())
//    ////    println(result)
//    //
//    //    val tmp = 0
//  }
}
