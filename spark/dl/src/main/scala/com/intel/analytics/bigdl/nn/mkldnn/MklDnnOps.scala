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
import com.intel.analytics.bigdl.mkl.MklDnn.EngineType

object MklDnnOps {

  private val engineType = EngineType.cpu

  def engineCreate(index : Long) : Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EngineCreate(engineType, index)
  }

  def engineDestroy(engine: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EngineDestroy(engine)
  }

  def streamCreate(streamKind: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamCreate(streamKind)
  }

  def streamSubmit(loc: Long, block: Int, primitives: Array[Long], length: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamSubmit(loc, block, primitives, length)
  }

  def streamWait(loc: Long, block: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamWait(loc, block)
  }

  def streamDestroy(loc: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamDestroy(loc)
  }

  def memoryDescInit(ndims: Int, dims: Array[Int], dataType: Int, dataFormat: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryDescInit(ndims, dims, dataType, dataFormat)
  }

  def memoryPrimitiveDescCreate(desc: Long, engine: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryPrimitiveDescCreate(desc, engine)
  }

  def memoryGetDataHandle(memory: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryGetDataHandle(memory)
  }

  def memorySetDataHandle(memory: Long, data: Array[Float]): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemorySetDataHandle(memory, data)
  }

  def memoryReleaseDataHandle(data: Array[Float], ptr: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryReleaseDataHandle(data, ptr)
  }

  def primitiveCreate(desc: Long, inputs: Array[Long], outputs: Array[Long]): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveCreate(desc, inputs, outputs)
  }

  def primitiveDescCreate(opDesc: Long, engine: Long, hingForwardPrimitiveDesc: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDescCreate(opDesc, engine, hingForwardPrimitiveDesc)
  }

  def primitiveDescDestroy(desc: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDescDestroy(desc)
  }

  def primitiveDestroy(primitive: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDestroy(primitive)
  }

//  def primitiveAt(primitive: Long, output_index: Int): Long = {
//    require(MklDnn.isLoaded, "mkldnn isn't loaded")
//    MklDnn.PrimitiveAt(primitive, output_index)
//  }

  def primitiveCreateForSubmit(desc: Long, inputs: Array[Long], length1: Int,
                               outputs: Array[Long], lenght2 : Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveCreateForSubmit(desc, inputs, length1, outputs, lenght2)
  }

  def eltwiseForwardDescInit(propKind: Int, algKind: Int, srcDesc: Long,
                             alpha: Float, beta: Float): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EltwiseForwardDescInit(propKind, algKind, srcDesc,
      alpha, beta)
  }

  def eltwiseBackwardDescInit(algKind: Int, diffDataDesc: Long, dataDesc: Long,
                              alpha: Float, beta: Float): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EltwiseBackwardDescInit(algKind, diffDataDesc, dataDesc,
      alpha, beta)
  }

  def initDataMemory(dim: Int, dims: Array[Int], memoryFormat: Int,
                     dataType: Int, engine: Long, data: Array[Float]): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    val prim_md = MklDnn.MemoryDescInit(dim, dims, dataType, memoryFormat)
    val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)

    val t1 = new Array[Long](1)
    val t2 = new Array[Long](1)
    val memory = MklDnn.PrimitiveCreate(user_pd, t1, t2)
    memorySetDataHandle(memory, data)
    primitiveDescDestroy(user_pd)

//    val dims = Array(1, 1, 2, 2)
//    val prim_md = MklDnn.MemoryDescInit(4, dims, MklDnn.DataType.f32, MklDnn.MemoryFormat.nchw)
//    val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
//    val t1 = new Array[Long](1)
//    val t2 = new Array[Long](1)
//    val relu_src_memory = MklDnn.PrimitiveCreate(user_pd, t1, t2)
//    MklDnnOps.memorySetDataHandle(relu_src_memory, data)
//    MklDnnOps.primitiveDescDestroy(user_pd)
//

    memory
  }

  def convForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, weights_desc: Long,
    bias_desc: Long, dst_desc: Long, strides: Array[Int],
    padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int): Long = {
      MklDnn.ConvForwardDescInit(prop_kind, alg_kind, src_desc, weights_desc,
        bias_desc, dst_desc, strides, padding_l, padding_r, padding_kind)
  }

  def convBackwardWeightsDescInit(alg_kind: Int, src_desc: Long, diff_weights_desc: Long,
    diff_bias_desc: Long, diff_dst_desc: Long, strides: Array[Int],
    padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int): Long = {
    MklDnn.ConvBackwardWeightsDescInit(alg_kind, src_desc, diff_weights_desc,
      diff_bias_desc, diff_dst_desc, strides, padding_l, padding_r, padding_kind)
  }

  def convBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
    diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
    padding_kind: Int): Long = {
    MklDnn.ConvBackwardDataDescInit(alg_kind, diff_src_desc, weights_desc,
      diff_dst_desc, strides, padding_l, padding_r, padding_kind)
  }
}
