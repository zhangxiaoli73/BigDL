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

import breeze.linalg.{*, product}
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.sql.catalyst.expressions.Conv
import spire.std.float

import scala.reflect.ClassTag

class ConvolutionDnn[T: ClassTag](
     val nInputPlane: Int,
     val nOutputPlane: Int,
     val kW: Int,
     val kH: Int,
     val dW: Int = 1,
     val dH: Int = 1,
     val padW: Int = 0,
     val padH: Int = 0,
     var adjW: Int = 0,
     var adjH: Int = 0
   )(implicit ev: TensorNumeric[T])
  extends TensorModule[Float] with Initializable {

  val convEngine = MklDnnOps.engineCreate(0)

  val weight: Tensor[Float] = Tensor[Float](nOutputPlane, nInputPlane, kH, kW)
  val bias: Tensor[Float] = Tensor[Float](nOutputPlane)
  val gradWeight: Tensor[Float] = Tensor[Float](nOutputPlane, nInputPlane, kH, kW)
  val gradBias: Tensor[Float] = Tensor[Float](nOutputPlane)

  val stdv = 1.0 / math.sqrt(kW * kH * nInputPlane)
  val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
  val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
  setInitMethod(wInit, bInit)

  val conv_weights_sizes = weight.size()
  val conv_bias_sizes = bias.size()
  val conv_strides = Array(dW, dH)
  val conv_padding = Array(padH, padW)

  // test
  var conv_src_memory : Long = 0L
  var conv_pd: Long = 0L
  var conv_weights_memory: Long = 0L
  var conv_bias_memory: Long = 0L

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.GP_IN_OUT_KW_KH)
    biasInitMethod.init(bias, VariableFormat.ONE_D)
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    val net_src_sizes = input.size()
    val batchSize = input.size(1)
    val conv_dst_sizes = Array(batchSize, nOutputPlane, 6, 6)

    conv_src_memory = MklDnnOps.initDataMemory(4, net_src_sizes, MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, convEngine, input.storage().array())
    conv_weights_memory = MklDnnOps.initDataMemory(4, conv_weights_sizes,
      MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, convEngine, weight.storage().array())
    conv_bias_memory = MklDnnOps.initDataMemory(1, conv_bias_sizes, MklDnn.MemoryFormat.x,
      MklDnn.DataType.f32, convEngine, bias.storage().array())

    /* create data descriptors for convolution w/ no specified format */
    // todo: for any format
    val conv_src_md = MklDnnOps.memoryDescInit(4, net_src_sizes,
      MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
    val conv_weights_md = MklDnnOps.memoryDescInit(4, conv_weights_sizes,
      MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
    val conv_bias_md = MklDnnOps.memoryDescInit(1, conv_bias_sizes,
      MklDnn.DataType.f32, MklDnn.MemoryFormat.x)
    val conv_dst_md = MklDnnOps.memoryDescInit(4, conv_dst_sizes,
      MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)


    /* create a convolution */
    val conv_desc = MklDnnOps.convForwardDescInit(MklDnn.PropKind.forward,
                      MklDnn.AlgKind.convolutionDirect, conv_src_md, conv_weights_md,
                      conv_bias_md, conv_dst_md, conv_strides, conv_padding,
                      conv_padding, MklDnn.PaddingKind.mkldnnPaddingZero)

    conv_pd = MklDnnOps.primitiveDescCreate(conv_desc, convEngine, 0L)

    // dst output
    output.resize(conv_dst_sizes)
    val arrData = output.storage().array()
    val conv_dst_memory = MklDnnOps.initDataMemory(4, conv_dst_sizes,
      MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, convEngine, arrData)

    /* create memory for dst data, we don't need to reorder it to user data */
    val conv_srcs = Array(conv_src_memory, conv_weights_memory, conv_bias_memory)
    val conv_dsts = Array(conv_dst_memory)

    /* finally create a convolution primitive */
    val conv = MklDnnOps.primitiveCreateForSubmit(conv_pd, conv_srcs, 3, conv_dsts, 1)

    /* build a simple net */
    val t1 = Array(conv)
    val reluStream = MklDnnOps.streamCreate(MklDnn.StreamType.eager)
    MklDnnOps.streamSubmit(reluStream, 1, t1, 1)
    MklDnnOps.streamWait(reluStream, 1)
    MklDnnOps.streamDestroy(reluStream)

    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {

    // bind memory
    gradInput.resizeAs(input)
    val conv_diff_src_memory = MklDnnOps.initDataMemory(4, input.size(), MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, convEngine, gradInput.storage().array())
    // gradOutput
    val conv_diff_dst_memory = MklDnnOps.initDataMemory(4, gradOutput.size(), MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, convEngine, gradOutput.storage().array())

//    val conv_weights_memory = MklDnnOps.initDataMemory(4, conv_weights_sizes,
//      MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, convEngine, weight.storage().array())

    // todo: support format any
    val conv_diff_src_md = MklDnnOps.memoryDescInit(4, input.size(), MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
    val conv_weights_md = MklDnnOps.memoryDescInit(4, conv_weights_sizes, MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
    val conv_diff_dst_md = MklDnnOps.memoryDescInit(4, gradOutput.size(), MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
    val conv_bias_md = MklDnnOps.memoryDescInit(1, conv_bias_sizes, MklDnn.DataType.f32, MklDnn.MemoryFormat.x)

    val conv_bwd_data_desc = MklDnnOps.convBackwardDataDescInit(
      MklDnn.AlgKind.convolutionDirect, conv_diff_src_md, conv_weights_md,
      conv_diff_dst_md, conv_strides, conv_padding,
      conv_padding, MklDnn.PaddingKind.mkldnnPaddingZero)

    val conv_bwd_weights_pd = MklDnnOps.primitiveDescCreate(conv_bwd_data_desc, convEngine, conv_pd)

    val conv_srcs = Array(conv_diff_dst_memory, conv_weights_memory, conv_src_memory)
    val conv_dsts = Array(conv_diff_src_memory)

    /* finally create a convolution primitive */
    val conv = MklDnnOps.primitiveCreateForSubmit(conv_bwd_weights_pd, conv_srcs, 3, conv_dsts, 1)

    /* build a simple net */
    val t1 = Array(conv)
    val reluStream = MklDnnOps.streamCreate(MklDnn.StreamType.eager)
    MklDnnOps.streamSubmit(reluStream, 1, t1, 1)
    MklDnnOps.streamWait(reluStream, 1)
    MklDnnOps.streamDestroy(reluStream)

    gradInput
  }

 override def accGradParameters(input: Tensor[Float], gradOutput: Tensor[Float]): Unit = {
   // bind memory
   val conv_diff_weights_memory = MklDnnOps.initDataMemory(4, conv_weights_sizes, MklDnn.MemoryFormat.nchw,
     MklDnn.DataType.f32, convEngine, gradWeight.storage().array())
   // gradOutput
   val conv_diff_dst_memory = MklDnnOps.initDataMemory(4, gradOutput.size(), MklDnn.MemoryFormat.nchw,
     MklDnn.DataType.f32, convEngine, gradOutput.storage().array())

   val conv_diff_bias_memory = MklDnnOps.initDataMemory(1, conv_bias_sizes, MklDnn.MemoryFormat.x,
     MklDnn.DataType.f32, convEngine, gradBias.storage().array())

   // todo: support format any
   val conv_bwd_src_md = MklDnnOps.memoryDescInit(4, input.size(), MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
   val conv_diff_weights_md = MklDnnOps.memoryDescInit(4, conv_weights_sizes, MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
   val conv_diff_bias_md = MklDnnOps.memoryDescInit(1, conv_bias_sizes, MklDnn.DataType.f32, MklDnn.MemoryFormat.x)
   val conv_diff_dst_md = MklDnnOps.memoryDescInit(4, gradOutput.size(), MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)


   val conv_bwd_weights_desc = MklDnnOps.convBackwardWeightsDescInit(
     MklDnn.AlgKind.convolutionDirect, conv_bwd_src_md, conv_diff_weights_md,
     conv_diff_bias_md, conv_diff_dst_md, conv_strides, conv_padding,
     conv_padding, MklDnn.PaddingKind.mkldnnPaddingZero)

   val conv_bwd_weights_pd = MklDnnOps.primitiveDescCreate(conv_bwd_weights_desc, convEngine, conv_pd)

   val conv_srcs = Array(conv_src_memory, conv_diff_dst_memory)
   val conv_dsts = Array(conv_diff_weights_memory, conv_diff_bias_memory)

   /* finally create a convolution primitive */
   val conv = MklDnnOps.primitiveCreateForSubmit(conv_bwd_weights_pd, conv_srcs, 2, conv_dsts, 2)

   /* build a simple net */
   val t1 = Array(conv)
   val reluStream = MklDnnOps.streamCreate(MklDnn.StreamType.eager)
   MklDnnOps.streamSubmit(reluStream, 1, t1, 1)
   MklDnnOps.streamWait(reluStream, 1)
   MklDnnOps.streamDestroy(reluStream)
 }
}

object Conv {
  def apply(
   nInputPlane: Int,
   nOutputPlane: Int,
   kW: Int,
   kH: Int,
   dW: Int = 1,
   dH: Int = 1,
   padW: Int = 0,
   padH: Int = 0,
   adjW: Int = 0,
   adjH: Int = 0): ConvolutionDnn[Float] = {
    new ConvolutionDnn[Float](nInputPlane, nOutputPlane, kW, kH, dW,
    dH, padW, padH, adjW, adjH)
  }
}
