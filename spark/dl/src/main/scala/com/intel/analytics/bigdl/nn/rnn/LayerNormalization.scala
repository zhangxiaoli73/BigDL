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
package com.intel.analytics.bigdl.nn.rnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
import org.dmg.pmml.True

import scala.reflect.ClassTag

class LayerNormalization[T: ClassTag](hidden_size: Int)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  private val epsilon = ev.fromType(1e-6)

//  private var weights = Tensor[T](hidden_size).fill(ev.one)
//  private var bias = Tensor[T](hidden_size).fill(ev.zero)

  private var weight = Tensor[T](T(-0.14037117, -0.16902402, -0.06451887,
    -0.5642037,   0.24212438,  0.44951588, -0.4296978,   0.423163))
  private var bias = Tensor[T](T(0.44111532, -0.06523705, -0.3474969,
    -0.08237404, -0.3565278,  -0.18157673, 0.4592312,  -0.36194998))

  var gradWeight = Tensor[T](hidden_size)
  var gradBias = Tensor[T](hidden_size)

  private var subMean = Tensor[T]()

  private var dimension = 2
  @transient
  private var _gradOutput: Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val inputX = input.toTensor[T].clone()
    val mean = inputX.mean(2).squeeze() // todo: other dimes??

    output = mean
    return output
//
    inputX.select(1, 1).sub(mean.valueAt(1))
    inputX.select(1, 2).sub(mean.valueAt(2))
    subMean = inputX.clone()
    if (output == null) output = Tensor[T]()
    output.toTensor[T].resizeAs(inputX).copy(inputX)
//    inputX.square()
//
//    var variance = inputX.mean(2).squeeze().add(epsilon)
//    variance = variance.sqrt().squeeze()
//    output.toTensor[T].select(1, 1).div(variance.valueAt(1)).cmul(weight).add(bias)
//    output.toTensor[T].select(1, 2).div(variance.valueAt(2)).cmul(weight).add(bias)

    output.toTensor[T].select(1, 1).cmul(weight).add(bias)
    output.toTensor[T].select(1, 2).cmul(weight).add(bias)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
//    val out1 = gradOutput.select(1, 1)
//    val out2 = gradOutput.select(1, 2)
//
//    gradInput.resizeAs(gradOutput)
//    gradInput.select(1, 1).addcmul(out1, weight)
//    gradInput.select(1, 2).addcmul(out2, weight)

    val size = input.size()
    size(dimension - 1) = 1
    if (!gradOutput.isContiguous()) {
      _gradOutput = gradOutput.clone().view(size)
    } else {
      _gradOutput = gradOutput.view(size)
    }
    gradInput.resizeAs(input)
    gradInput.copy(_gradOutput.expandAs(input))
    gradInput.div(ev.fromType(input.size(dimension)))

    gradInput
  }
}
