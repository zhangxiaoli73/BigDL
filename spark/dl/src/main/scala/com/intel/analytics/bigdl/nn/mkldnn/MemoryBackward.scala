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

import breeze.linalg
import breeze.linalg.dim
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, TensorModule}
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
  * when from mkldnn layer to bigdl layer, there need to do reorder for input or gradOutput
  */
class MemoryBackward(outputFormat: Int = MklDnn.MemoryFormat.nchw) extends TensorModule[Float] {

  @transient
  private var engine: Long = 0L
  @transient
  private var stream: Long = 0L
  @transient
  private var update_primitive: Boolean = true
  @transient
  private var inputElement : Int = 0
  @transient
  private var src_memory: Long = 0L
  @transient
  private var src_memory2: Long = 0L
  @transient
  private var dst_memory: Long = 0L
  @transient
  private var dst_memory2: Long = 0L
  @transient
  private var gradInput_memory: Long = 0L
  @transient
  private var gradOutput_memory: Long = 0L

  override val isMklDnnModel: Boolean = true

  val stream_fwd = new ArrayBuffer[Long]
  val stream_bwd = new ArrayBuffer[Long]

  private val dataType = MklDnn.DataType.f32

  private var internal_inputFormat = 0

  require(outputFormat != MklDnn.MemoryFormat.any,
    "output format in MemoryReOrder should not be any")

  // convert input from input format to output format
  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()
    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    if (inputElement != input.nElement()) {
      update_primitive = true
      inputElement = input.nElement()
    } else {
      update_primitive = false
    }

    output = input
    output
  }

  // convert gradOutput from output format to input format
  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()

    if (update_primitive) {
      val sizes = gradOutput.size()
      val dim = gradOutput.dim()

      internal_inputFormat = gradOutput.getFormat()
      val src_pd = gradOutput.getPrimitiveDesc()

      val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, internal_inputFormat)
      val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
      dst_memory2 = MklDnn.PrimitiveCreate0(user_pd)

      val res = MklDnnOps.prepareReorder(dst_memory2, src_pd, false)
      // val reorder_primitive = res._1
      src_memory2 = res._2
      if (src_memory2 != 0L) {
        // todo: output with Dense Tensor
        gradInput = MklDnnTensor[Float](gradOutput.size())
        gradInput.setPrimitiveDesc(user_pd)
      }

      stream_bwd.clear()
      stream_bwd.append(res._1)
    }

    /* build a simple net */
    if (src_memory2 != 0L) {
      val memoryPrimitives = Array(src_memory2, dst_memory2)
      val buffer = Array(gradOutput, gradInput)
      MklDnnOps.streamSubmit(stream, 1, stream_bwd.toArray, 1, memoryPrimitives, buffer)
    } else {
      gradInput = gradOutput
    }

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, gradOutput.getFormat(), gradInput.getFormat())
    }
    gradInput
  }
}

object MemoryBackward {
  def apply[T: ClassTag](outputFormat: Int = MklDnn.MemoryFormat.nchw): MemoryBackward = {
    new MemoryBackward(outputFormat)
  }
}
