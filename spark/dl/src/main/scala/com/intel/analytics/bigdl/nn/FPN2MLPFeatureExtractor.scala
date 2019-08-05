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

import breeze.linalg.dim
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.dmg.pmml.{False, True}

class FPN2MLPFeatureExtractor(in_channels: Int, resolution: Int,
  scales: Array[Float], sampling_ratio: Float, representation_size: Int, use_gn: Boolean = false)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {

//  val pooler = Pooler((resolution, resolution), scales, sampling_ratio)
  val input_size = in_channels * math.pow(resolution, 2)
  val fc6 = makeFC(input_size.toInt, representation_size, use_gn)
  val fc7 = makeFC(representation_size, representation_size, use_gn)

  val model = Sequential[Float]()
            .add(fc6)
            .add(ReLU[Float]())
            .add(fc7).add(ReLU[Float]())

  /**
   * Caffe2 implementation uses XavierFill, which in fact corresponds to kaiming_uniform_ in PyTorch
   * @param dim_in
   * @param hidden_dim
   * @param use_gn
   */
  private def makeFC(dim_in: Int, hidden_dim: Int, use_gn: Boolean = false): Module[Float] = {
    val fc = if (use_gn) { // not support
      val fc = Linear[Float](dim_in, hidden_dim, withBias = false)
      // todo: use XavierFill initial
      fc
    } else {
      val fc = Linear[Float](dim_in, hidden_dim, withBias = false)
      // todo: use XavierFill initial
      fc.bias.fill(ev.zero)
      fc
    }
    fc.asInstanceOf[Module[Float]]
  }

  private def groupNorm(out_channels: Int, affine : Boolean = true, divisor: Int = 1,
    eps: Double = 1e-5, num_groups: Int, dim_per_gp: Int): Unit = {
    val tmp_out_channels = out_channels % divisor
    val tmp_dim_per_gp = dim_per_gp % divisor
    val tmp_num_groups = num_groups % divisor

    val group_gn = getGroupGn(out_channels, dim_per_gp, num_groups)

//    return torch.nn.GroupNorm(
//      get_group_gn(out_channels, dim_per_gp, num_groups),
//      out_channels,
//      eps,
//      affine
//    )
  }

  // get number of groups used by GroupNorm, based on number of channels.
  private def getGroupGn(dim: Int, dim_per_gp: Int, num_groups: Int): Int = {
    require(dim_per_gp == -1 || num_groups == -1, "GroupNorm: can only specify G or C/G.")
    if (dim_per_gp > 0) {
      require(dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp))
      dim % dim_per_gp
    } else {
      require(dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups))
      num_groups
    }
  }

  // input contains:
  // 1th features, and 2th proposals
  override def updateOutput(input: Table): Tensor[Float] = {
    val features = input[Tensor[Float]](1)
    val proposals = input[Tensor[Float]](2)

//    x = self.pooler(x, proposals)
//    x = x.view(x.size(0), -1)

    model.forward(features)
    output = model.output.toTensor[Float]
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput
  }

}
