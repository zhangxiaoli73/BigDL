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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, T, Table}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import scala.reflect.ClassTag
import scala.util.Random

class SharedGradInputSpec extends FlatSpec with Matchers {


  "Inception+bn" should "generate correct output" in {
    Random.setSeed(3)
    val input = Tensor[Float](4, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => Random.nextInt(1000))

    val seed = 100
    RNG.setSeed(seed)

    val model = Inception.getModel[Float](1000, "inception-bn")
    val sharedModel = Inception.getModel[Float](1000, "inception-bn")
    ShareGradInput.shareConvolution(sharedModel)
        ShareGradInput.shareGradInput(sharedModel)

    val (weights, grad) = model.getParameters()
    val (w, g) = sharedModel.getParameters()
    w.copy(weights)

    val output1 = model.forward(input).toTensor[Float]
    val output2 = sharedModel.forward(input).toTensor[Float]
    output1 should be (output2)
    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)
//    model.backward(input, gradOutput)
    val gradInput1 = model.backward(input, gradOutput)
//    sharedModel.backward(input, gradOutput)
    val gradInput2 = sharedModel.backward(input, gradOutput)
    gradInput1 should be (gradInput2)

    File.save(model, "/tmp/unshared.inception", true)
    File.save(sharedModel, "/tmp/shared.inception2", true)

    println()

  }
}
