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

import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{Squeeze, mkldnn}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

class BigDL2DnnWrapperSpec extends BigDLSpecHelper {

  // from nhwc -> nchw
  def shapeToNCHW(shape: Array[Int]): Array[Int] = {
   Array(shape(0), shape(3), shape(1), shape(2))
  }

  def model(inputShape: Array[Int]) : MklDnnContainer = {
    val model1 = mkldnn.Sequential()

    model1.add(mkldnn.Input(inputShape, Memory.Format.nhwc))

    val s = Squeeze[Float](Array(2, 3), true).
      asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]

    model1.add(BigDL2DnnWrapper(s, ""))

    model1
  }

  def wrapConv(inputShape: Array[Int], gradOutShape : Array[Int]) : MklDnnContainer = {
    import com.intel.analytics.bigdl.nn
    RNG.setSeed(100)

    val model1 = mkldnn.Sequential()

    model1.add(mkldnn.Input(inputShape, Memory.Format.nhwc).setName("input"))

    // conv with NCHW
    val s = nn.SpatialConvolution[Float](3, 32, 5, 5, 1, 1).
      asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]
//    s.getParameters()._1.fill(0.0001f)
//    s.getParameters()._2.fill(0.0001f)

//    val p = s.getParametersTable()
//    p.get[Tensor[Float]]("weight").get.copy(initWeight)
//    p.get[Tensor[Float]]("bias").get.copy(initBias)

    model1.add(BigDL2DnnWrapper(s, "").setName("wrapper"))

    model1.add(ReorderMemory(
      inputFormat = null, // HeapData(Array(4, 32, 3, 3), Memory.Format.nchw),
      outputFormat = null, // HeapData(Array(4, 3, 3, 32), Memory.Format.nhwc),
      gradInputFormat = HeapData(Array(4, 32, 3, 3), Memory.Format.nchw),
      gradOutputFomat = HeapData(gradOutShape, Memory.Format.nhwc)).setName("test"))
    model1
  }

   "wrapper squeeze" should "be correct" in {
     val in = Tensor[Float](2, 1, 1, 3).rand()
     val inNHWC = in.transpose(2, 4).transpose(3, 4).contiguous().clone()

     val s = Squeeze[Float](Array(2, 3), true).
       asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]

     val w = model(inNHWC.size()) // BigDL2DnnWrapper(s, "")

     w.compile(Phase.InferencePhase)

     val out = w.forward(inNHWC)

     val tm = 0
   }

  "wrapper conv" should "be correct" in {
    val nIn = 3
    val nOut = 32
    val kH = 5
    val kW = 5
    RNG.setSeed(100)
    val in = Tensor[Float](4, 3, 7, 7).apply1(_ => RNG.uniform(0.1, 1).toFloat)
    val inNHWC = in.transpose(2, 3).transpose(3, 4).contiguous().clone()

    val gradOutput = Tensor[Float](4, 32, 3, 3).apply1(_ => RNG.uniform(0.1, 1).toFloat)
    val gradOutputNHWC = gradOutput.transpose(2, 3).transpose(3, 4).contiguous().clone()


    RNG.setSeed(100)
    val model = nn.SpatialConvolution[Float](3, 32, 5, 5, 1, 1).
      asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]


    // wrapper
    val wrapperModel = wrapConv(inNHWC.size(), gradOutputNHWC.size())
    wrapperModel.compile(Phase.TrainingPhase)

    val wrapperOut = wrapperModel.forward(inNHWC)
    val out = model.forward(in) // size 4, 32, 3, 3

    wrapperOut.equals(out) should be(true)

    // for backward
    val grad = model.backward(in, gradOutput)
    val t1 = grad.clone()
    val t2 = grad.clone()


    val wrapperGrad = wrapperModel.backward(inNHWC, gradOutputNHWC)
    val gradNHWC = grad.transpose(2, 3).transpose(3, 4).contiguous().clone()

    gradNHWC should be(wrapperGrad)

    println("done")
  }
}
