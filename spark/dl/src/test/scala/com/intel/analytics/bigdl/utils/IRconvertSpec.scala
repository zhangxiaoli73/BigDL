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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.nn.{Graph, Reshape, StaticGraph}
import com.intel.analytics.bigdl.nn.mkldnn.DnnGraph
import com.intel.analytics.bigdl.utils.mkldnn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

class IRconvertSpec extends BigDLSpecHelper {

  def modelIR(inputFormats: Int = Memory.Format.nchw,
              outputFormats: Int = Memory.Format.nchw): IRGraph[Float] = {
    val conv1 = Node(IRElement[Float]("", IRSpatialConvolution[Float](1, 20, 5, 5)))
    val bn1 = Node(IRElement[Float]("", IRSpatialBatchNormalization[Float](20)))
    val pool1 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val conv2 = Node(IRElement[Float]("", IRSpatialConvolution[Float](20, 50, 5, 5)))
    val pool2 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))

    conv1 -> bn1 -> pool1 -> conv2 -> pool2
    val output = pool2
    IRGraph(Array(conv1), Array(output), inputFormats = inputFormats, outputFormats = outputFormats)
  }

  def modelIR2(inputFormats: Int = Memory.Format.nchw,
              outputFormats: Int = Memory.Format.nc): IRGraph[Float] = {
    val conv1 = Node(IRElement[Float]("", IRSpatialConvolution[Float](1, 20, 5, 5)))
    val pool1 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val conv2 = Node(IRElement[Float]("", IRSpatialConvolution[Float](20, 50, 5, 5)))
    val pool2 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val reshape = Node(IRElement("", IRReshape[Float](Array(50*4*4))))
    val linear = Node(IRElement("", IRLinear[Float](50 * 4 * 4, 500)))
    val relu = Node(IRElement("", IRReLU[Float]()))
    val fc2 = Node(IRElement("", IRLinear[Float](500, 10)))

    conv1 -> pool1 -> conv2 -> pool2 ->
    reshape -> linear -> relu -> fc2
    val output = fc2

    IRGraph(Array(conv1), Array(output), inputFormats = inputFormats, outputFormats = outputFormats)
  }

  def modelBlas(format: DataFormat = DataFormat("NCHW")) : Module[Float] = {
    val conv1 = nn.SpatialConvolution(1, 20, 5, 5, format = format).inputs()
    val pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).setName("pool").inputs(conv1)
    val conv2 = nn.SpatialConvolution(20, 50, 5, 5, format = format).inputs(pool1)
    val pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).inputs(conv2)
    val reshape = nn.Reshape(Array(50 * 4 * 4)).inputs(pool2)
    val fc = nn.Linear(50 * 4 * 4, 500).inputs(reshape)
    val relu = nn.ReLU().setName("relu1").inputs(fc)
    val fc2 = nn.Linear(500, 10).setName("ip2").inputs(relu)
    Graph(conv1, fc2)
  }

  "Convert Blas with NHWC to Dnn" should "be correct" in {
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val inpuNHWC = input.transpose(2, 3).transpose(3, 4).contiguous().clone()
    val gradOutput = Tensor[Float](2, 10).rand()

    val blas = modelBlas(DataFormat("NHWC")).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph(inputFormats = Memory.Format.nhwc)

    System.setProperty("bigdl.engineType", "mklblas")
    irBlas.build()
    val outBlas = irBlas.forward(inpuNHWC)
    val gradInputBlas = irBlas.backward(inpuNHWC, gradOutput)

    System.setProperty("bigdl.engineType", "mkldnn")
    irBlas.build()
    val outDnn = irBlas.forward(input)
    val gradInputDnn = irBlas.backward(input, gradOutput)

    val outBlas1 = blas.forward(inpuNHWC)
    val gradInputBlas1 = blas.backward(inpuNHWC, gradOutput)

    outDnn should be(outBlas)
    gradInputBlas should be(gradInputDnn)

  }

  "Convert Blas with NCHW to Dnn" should "be correct" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val gradOutput = Tensor[Float](2, 10).rand()

    val blas = modelBlas().asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph()

    val convert = new IRConverter[Float](irBlas)
    irBlas.graph = convert.toBlasGraph()
    val outBlas = irBlas.forward(input)
    val gradInputBlas = irBlas.backward(input, gradOutput)

    System.setProperty("bigdl.engineType", "mkldnn")
    irBlas.build()
    irBlas.graph = convert.toBlasGraph()
    val outDnn = irBlas.forward(input)
    val gradInputDnn = irBlas.backward(input, gradOutput)

    val outBlas1 = blas.forward(input)
    val gradInputBlas1 = blas.backward(input, gradOutput)

    outDnn should be(outBlas)
    gradInputBlas should be(gradInputDnn)

  }

  "Convert IRgraph to Dnn or Blas Graph" should "be correct" in {
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val gradOutput = Tensor[Float](2, 50, 4, 4).rand()

    RandomGenerator.RNG.setSeed(1000)
    System.setProperty("bigdl.engineType", "mklblas")
    val irBlas = modelIR()
    irBlas.build()
    val outBlas = irBlas.forward(input)
    val gradInputBlas = irBlas.backward(input, gradOutput)

    RandomGenerator.RNG.setSeed(1000)
    System.setProperty("bigdl.engineType", "mkldnn")
    val irDnn = modelIR()
    irDnn.build()
    val outDnn = irDnn.forward(input)
    val gradInputDnn = irDnn.backward(input, gradOutput)

    outDnn should be(outBlas)
    gradInputBlas should be(gradInputDnn)
  }

  "Convert IRgraph to Dnn or Blas Graph with 2 dimentions output" should "be correct" in {
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val gradOutput = Tensor[Float](2, 10).rand()

    RandomGenerator.RNG.setSeed(1000)
    System.setProperty("bigdl.engineType", "mklblas")
    val irBlas = modelIR2()
    irBlas.build()
    val outBlas = irBlas.forward(input)
    val gradInputBlas = irBlas.backward(input, gradOutput)

    RandomGenerator.RNG.setSeed(1000)
    System.setProperty("bigdl.engineType", "mkldnn")
    val irDnn = modelIR2()
    irDnn.build()
    val outDnn = irDnn.forward(input)
    val gradInputDnn = irDnn.backward(input, gradOutput)

    outDnn should be(outBlas)
    gradInputBlas should be(gradInputDnn)
  }
}
