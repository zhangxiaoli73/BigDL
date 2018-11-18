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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.mkldnn.{IRElement, IRSpatialMaxPooling}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer._

import scala.reflect.{ClassTag, ManifestFactory}
import scala.reflect.runtime._
import com.intel.analytics.bigdl.{Module, nn}

class ReflectionUtilsSpec extends BigDLSpecHelper {

  "test SpatialConvolution reflection" should "be right" in {
    val model1 = nn.SpatialConvolution(3, 32, 5, 5)
    val className = "com.intel.analytics.bigdl.utils.mkldnn.IRSpatialConvolution"
    val cls = Class.forName(className)
    val ir = ReflectionUtils.reflectToIR(model1, cls)
    val cls2 = Class.forName(
      "com.intel.analytics.bigdl.nn.SpatialConvolution")
    val model2 = ReflectionUtils.reflectFromIR(ir, cls2)

    val input = Tensor[Float](4, 3, 7, 7).rand()
    val gradOutput = Tensor[Float](4, 32, 5, 5).rand()

    val out1 = model1.forward(input)
    val out2 = model2.forward(input)


    val grad1 = model1.backward(input, gradOutput)
    val grad2 = model2.backward(input, gradOutput)


    println("done")
  }

  "test BatchNorm reflection" should "be right" in {
    val model1 = nn.SpatialBatchNormalization(3)
    val className = "com.intel.analytics.bigdl.utils.mkldnn.IRSpatialBatchNormalization"
    val cls = Class.forName(className)
    val ir = ReflectionUtils.reflectToIR(model1, cls)
    val cls2 = Class.forName("com.intel.analytics.bigdl.nn.SpatialBatchNormalization")
    val modelBlas = ReflectionUtils.reflectFromIR(ir, cls2)

    val cls3 = Class.forName("com.intel.analytics.bigdl.nn.mkldnn.SpatialBatchNormalization")
    val modelDnn = ReflectionUtils.reflectFromIR(ir, cls3)

    val input = Tensor[Float](16, 3, 4, 4).rand()
    val gradOutput = Tensor[Float](16, 3, 4, 4).rand()

    val out1 = model1.forward(input)
    val out2 = modelBlas.forward(input)


    val grad1 = model1.backward(input, gradOutput)
    val grad2 = modelBlas.backward(input, gradOutput)


    println("done")
  }



}
