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

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ReLUDnn[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[Float] {

  val reluEngine = MklDnnOps.engineCreate(0)
  var relu_src_md: Long = 0L

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    val src_size = input.size()
    val dst_size = src_size
    output.resizeAs(input)
    val inputBuffer = input.contiguous()
    val inputArray = inputBuffer.storage().array()

    // todo: need to release
    val relu_src_memory = MklDnnOps.initDataMemory(4, src_size, MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, reluEngine, inputArray)

//    val prim_md = MklDnn.MemoryDescInit(4, src_size, MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
//    val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, reluEngine)
//    val t1 = new Array[Long](1)
//    val t2 = new Array[Long](1)
//    val relu_src_memory = MklDnn.PrimitiveCreate(user_pd, t1, t2)
//    MklDnnOps.memorySetDataHandle(relu_src_memory, inputArray)
//    MklDnnOps.primitiveDescDestroy(user_pd)

    relu_src_md = MklDnnOps.memoryDescInit(4, src_size, MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
    val relu_desc = MklDnnOps.eltwiseForwardDescInit(
      MklDnn.PropKind.forward, MklDnn.AlgKind.eltwiseRelu, relu_src_md, 0, 0)


    val relu_pd = MklDnnOps.primitiveDescCreate(relu_desc, reluEngine, 0L)

    /* create relu dst memory primitive */
    val relu_dst_memory = MklDnnOps.initDataMemory(4, dst_size, MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, reluEngine, output.storage().array())

//    val prim_md2 = MklDnn.MemoryDescInit(4, dst_size, MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
//    val user_pd2 = MklDnn.MemoryPrimitiveDescCreate(prim_md2, reluEngine)
//    val t12 = new Array[Long](1)
//    val t22 = new Array[Long](1)
//    val relu_dst_memory = MklDnn.PrimitiveCreate(user_pd2, t12, t22)
//    MklDnnOps.memorySetDataHandle(relu_dst_memory, output.storage().array())
//    MklDnnOps.primitiveDescDestroy(user_pd2)

//
//    /* finally create a relu primitive */
    val relu_srcs = Array(relu_src_memory)
    val relu_dsts = Array(relu_dst_memory)
    // println("scala " + relu_pd)
    val relu = MklDnnOps.primitiveCreateForSubmit(relu_pd, relu_srcs, 1, relu_dsts, 1)

    // println("here " + relu)
    val t = Array(relu)
    val reluStream = MklDnnOps.streamCreate(MklDnn.StreamType.eager)
    MklDnnOps.streamSubmit(reluStream, 1, t, 1)
    MklDnnOps.streamWait(reluStream, 1)
    MklDnnOps.streamDestroy(reluStream)
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    /* ... user diff_data ...*/
    val src_size = input.size()
    val dst_size = src_size
    val inputBuffer = input.contiguous()
    val inputArray = inputBuffer.storage().array()

    val relu_src_memory = MklDnnOps.initDataMemory(4, src_size, MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, reluEngine, inputArray)

    gradInput.resizeAs(input)
    val relu_diff_dst_memory = MklDnnOps.initDataMemory(4, gradOutput.size(), MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, reluEngine, gradOutput.storage().array())

    /* Backward relu */
    // const mkldnn_memory_desc_t *relu_diff_dst_md = mkldnn_primitive_desc_query_memory_d(relu_dst_pd);
    // mkldnn_memory_desc_t *relu_diff_dst_md = malloc(sizeof(mkldnn_memory_desc_t));
    // mkldnn_memory_desc_init(relu_diff_dst_md, 4, relu_dst_sizes, mkldnn_f32, mkldnn_nchw);
    // relu_diff_dst_md = (mkldnn_memory_desc_t *)jni_MemoryDescInit(4, relu_dst_sizes, mkldnn_f32, mkldnn_nchw);

    val relu_diff_dst_md = MklDnn.MemoryDescInit(4, src_size, MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)

    /* create backward relu descriptor */
    // mkldnn_eltwise_desc_t *relu_bwd_desc;
    // CHECK(mkldnn_eltwise_backward_desc_init(&relu_bwd_desc, mkldnn_eltwise_relu, relu_diff_dst_md, relu_src_md, negative_slope, 0));
    // relu_bwd_desc = (mkldnn_eltwise_desc_t *)jni_EltwiseBackwardDescInit(mkldnn_eltwise_relu, (long)relu_diff_dst_md, (long)relu_src_md, negative_slope, 0);

    val relu_bwd_desc = MklDnnOps.eltwiseBackwardDescInit(MklDnn.AlgKind.eltwiseRelu, relu_diff_dst_md, relu_src_md, 0, 0)
//
//    mkldnn_primitive_desc_t *relu_bwd_pd;
//    // CHECK(mkldnn_primitive_desc_create(&relu_bwd_pd, &relu_bwd_desc, *engine, *relu_pd));
//    relu_bwd_pd = (mkldnn_primitive_desc_t *)jni_PrimitiveDescCreate((long)relu_bwd_desc, *engine, (long)relu_pd);

    val relu_bwd_pd = MklDnnOps.primitiveDescCreate(relu_bwd_desc, reluEngine, 0L)

    /* create memory primities for relu diff src */
//    float *relu_diff_src_buffer = (float *)aligned_malloc(product(relu_src_sizes, 4)*sizeof(float), 64);
//    memset(relu_diff_src_buffer, 0, product(relu_src_sizes, 4)*sizeof(float));
//
//    mkldnn_primitive_t *relu_diff_src_memory = init_data_memory(4, relu_src_sizes, mkldnn_nchw, mkldnn_f32, *engine, relu_diff_src_buffer, NULL);
    val relu_diff_src_memory = MklDnnOps.initDataMemory(4, gradInput.size(),
      MklDnn.MemoryFormat.nchw, MklDnn.DataType.f32, reluEngine, gradInput.storage().array())

//    mkldnn_primitive_at_t relu_diff_dsts[] = { mkldnn_primitive_at(*relu_src_memory, 0),
//      mkldnn_primitive_at(*relu_diff_dst_memory, 0) };
//
//    const_mkldnn_primitive_t relu_diff_srcs[] = { *relu_diff_src_memory };

    val relu_diff_dsts = Array(relu_src_memory, relu_diff_dst_memory)
    val relu_diff_srcs = Array(relu_diff_src_memory)

    // println("scala backward " + relu_bwd_pd)
    val relu_bwd = MklDnnOps.primitiveCreateForSubmit(relu_bwd_pd, relu_diff_dsts, 2, relu_diff_srcs, 1)

    // println("here backward" + relu_bwd)
    val t1 = Array(relu_bwd)
    val reluStream = MklDnnOps.streamCreate(MklDnn.StreamType.eager)
    MklDnnOps.streamSubmit(reluStream, 1, t1, 1)
    MklDnnOps.streamWait(reluStream, 1)
    MklDnnOps.streamDestroy(reluStream)

    gradInput
  }
}

object ReLUDnn {
  def apply[T: ClassTag](ip: Boolean = false)(implicit ev: TensorNumeric[T]): ReLUDnn[T] = {
    new ReLUDnn[T](ip)
  }
}
