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

  def wrapSqueeze(inputShape: Array[Int]) : MklDnnContainer = {
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

    val s = nn.SpatialConvolution[Float](3, 32, 5, 5, 1, 1).
      asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]

    model1.add(BigDL2DnnWrapper(s, "").setName("wrapper"))
    model1.add(ReorderMemory(inputFormat = null, outputFormat = null,
      gradInputFormat = null,
      gradOutputFomat = HeapData(gradOutShape, Memory.Format.nhwc)).setName("test"))
    model1
  }

   "wrapper squeeze" should "be correct" in {
     val in = Tensor[Float](2, 1, 1, 3).rand()
     val inNHWC = in.transpose(2, 4).transpose(3, 4).contiguous().clone()

     val model = Squeeze[Float](Array(2, 3), true).
       asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]

     val wrapperModel = wrapSqueeze(inNHWC.size())
     wrapperModel.compile(Phase.InferencePhase)

     val outWrapper = wrapperModel.forward(inNHWC)
     val out = model.forward(in)

     outWrapper should be(out)
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
    val out = model.forward(in)

    wrapperOut.equals(out) should be(true)

    // for backward
    val grad = model.backward(in, gradOutput)
    val wrapperGrad = wrapperModel.backward(inNHWC, gradOutputNHWC)
    val gradNHWC = grad.transpose(2, 3).transpose(3, 4).contiguous().clone()

    gradNHWC should be(wrapperGrad)
  }
}
