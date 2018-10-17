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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}


class DnnGraphSpec extends FlatSpec with Matchers {
  def nonDnnGraph(): Module[Float] = {
    val input = nn.Identity[Float]().inputs()
    val conv1 = nn.SpatialConvolution[Float](1, 20, 5, 5).setName("conv1").inputs(input)
    val relu1 = nn.ReLU[Float]().setName("relu1").inputs(conv1)
    val tanh1 = nn.SpatialBatchNormalization[Float](20).inputs(relu1)
//    val relu2 = nn.ReLU[Float]().setName("relu2").inputs(tanh1)
//    val pool1 = nn.SpatialMaxPooling[Float](2, 2, 2, 2).inputs(relu2)
//    val conv2 = nn.SpatialConvolution[Float](20, 50, 5, 5).setName("conv2").inputs(pool1)
//    val pool2 = nn.SpatialMaxPooling[Float](2, 2, 2, 2).inputs(conv2)
//    val reshpe = nn.Reshape[Float](Array(50 * 4 * 4)).inputs(pool2)
//    val fc1 = nn.Linear[Float](50 * 4 * 4, 500).setName("fc1").inputs(reshpe)
//    val tanh3 = nn.ReLU[Float]().inputs(fc1)
//    val fc2 = nn.Linear[Float](500, 10).setName("fc2").inputs(tanh3)

    val output = tanh1
    Graph(input, output)
  }

  def dnnSequential(): mkldnn.Sequential = {
    val model = mkldnn.Sequential()
      .add(mkldnn.SpatialConvolution(1, 20, 5, 5).setName("conv1"))
      .add(mkldnn.ReLU().setName("relu1"))
      .add(mkldnn.SpatialBatchNormalization(20))
    model
  }

  def nonDnnSequential(classNum: Int): Module[Float] = {
    val input = Identity[Float]().inputs()
    val conv1 = SpatialConvolution(1, 20, 5, 5).setName("conv1").inputs(input)
    val relu1 = ReLU().setName("relu1").inputs(conv1)
    // val relu2 = ReLU().setName("relu2").inputs(relu1)
    //    val tanh1 = SpatialBatchNormalization(20).inputs(relu1)
    //    val relu2 = tanh1 // ReLU().setName("relu2").inputs(tanh1)
    //    val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu2)
    //    val conv2 = SpatialConvolution(20, 50, 5, 5).setName("conv2").inputs(pool1)
    //    val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(conv2)
    //    val reshpe = Reshape(Array(50 * 4 * 4)).inputs(pool2)
    //    val fc1 = Linear(50 * 4 * 4, 500).setName("fc1").inputs(reshpe)
    //    val tanh3 = ReLU().inputs(fc1)
    //    val output = Linear(500, 10).setName("fc2").inputs(tanh3)

    val output = relu1
    Graph(input, output)
  }

  "Non Dnn graph model convert to dnnGraph" should "be correct" in {
    val batchSize = 4
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 1, 28, 28)

    RNG.setSeed(seed)
    val model1 = nonDnnGraph()
    RNG.setSeed(seed)
    val model2 = nonDnnGraph()
    val dnnModle = model2.toDnnModule()
    RNG.setSeed(seed)
    val model3 = dnnSequential()

    dnnModle.asInstanceOf[DnnGraph[Float]]
      .compile(TrainingPhase, Array(HeapData(inputShape, inputFormat)))
    model3.compile(TrainingPhase, Array(HeapData(inputShape, inputFormat)))

    val input = Tensor[Float](Array(batchSize, 1, 28, 28)).fill(0.1f) // rand()// rand()
    val output = model1.forward(input).toTensor[Float]

    val gradOutput = Tensor[Float]().resizeAs(output).fill(1f)

    val gradInput = model1.backward(input, gradOutput).toTensor[Float]
    val outputDnn = DnnTools.dense(dnnModle.forward(input)).toTensor[Float]
    val gradInputDnn = DnnTools.dense(dnnModle.backward(input, gradOutput)).toTensor[Float]

    val out = DnnTools.dense(model3.forward(input))
    val grad = DnnTools.dense(model3.backward(input, gradOutput))


    //    println(gradInput)
    //    println(gradInputDnn)
    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val t = 0
  }

  "Non Dnn sequential convert to dnnGraph" should "be correct" in {


  }

  "Dnn model convert to dnnGraph" should "be correct" in {


  }

  "Dnn resnet model convert to dnnGraph" should "be correct" in {


  }
}