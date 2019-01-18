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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

/**
 * `Blob` contains two tensors, dense and native, which are a map of each other.
 * It's used in the layer which contains weights. For the weight, we should sync the
 * dense tensor to native tensor before `submit`. For the gradient, we should sync the native
 * tensor to dense tensor after `submit`.
 *
 * The `setMemoryData` requires the elements number should be consistent. If the shape is not,
 * it will reshape first.
 *
 * The Blob has another attribute `_memoryData` and will not be determined when the blob created.
 * It can be determined when we initialize the primitives.
 *
 * @param _size the shape of Tensor, such as Array(4, 3, 224, 224)
 */
private[mkldnn] class Blob(_size: Array[Int]) extends Serializable {
  val dense: Tensor[Float] = Tensor[Float](_size)

  @transient var native: DnnTensor[Float] = _
  @transient private var _heapData: HeapData = null
  @transient private var _nativeData: NativeData = null
  @transient private var _reorderForward: ReorderMemory = null
  @transient private var _reorderBackward: ReorderMemory = null
  @transient private var _memoryData: MemoryData = _

  /**
   * it will copy the dense tensor to native tensor before `submit` reads the native tensor
   */
  def syncToNative(): Unit = {
    if (_reorderForward == null) {
      native.copy(dense)
    } else {
      _reorderForward.forward(this.dense).asInstanceOf[DnnTensor[Float]]
    }
  }

  /**
   * it will copy the native tensor to dense tensor after `submit` updates the native tensor
   */
  def syncToHeap(): Unit = {
    if (_reorderBackward == null) {
      dense.copy(native)
    } else {
      _reorderBackward.forward(this.native).asInstanceOf[Tensor[Float]]
    }
  }

  /**
   * MemoryData relevant of Native Tensor. The shape should be the same as `size` of Blob.
   * We can't only reserve the `layout` in MemoryData. Because for convolution,
   * we should reserve the whole MemoryData including `desc`, `desc primitive` and `primitive`.
   *
   * @param memoryData memory data you want.
   */
  def setMemoryData(memoryData: MemoryData): Unit = {
    require(_memoryData == null, "You should only set once")
    require(size().product == memoryData.shape.product, s"You may assign wrong layout")

    // we should resize the tensor. Because sometimes, weight of Linear will has 4-D, where
    // the last 2 dims is 1. we should reisze it. It will not allocate a new storage because of
    // the same size.
    List(native, dense).foreach(_.resize(memoryData.shape))
    _memoryData = memoryData
  }

  def memoryData(): MemoryData = {
    require(_memoryData != null, "You should setMemoryData first")
    _heapData
  }

  def isMemoryDataSet(): Boolean = {
    if (_memoryData == null) {
      false
    } else {
      true
    }
  }

  def setMemoryData(dense: HeapData, native: NativeData, runtime: MklDnnRuntime): Unit = {
    _heapData = dense
    _nativeData = native

    _reorderForward = ReorderMemory(_nativeData)
    _reorderForward.setRuntime(runtime)
    _reorderForward.initFwdPrimitives(Array(_heapData), InferencePhase)

    this.native = _reorderForward.updateOutput(this.dense).asInstanceOf[DnnTensor[Float]]

    _reorderBackward = ReorderMemory(_heapData)
    _reorderBackward.setRuntime(runtime)
    _reorderBackward.initFwdPrimitives(Array(_nativeData), InferencePhase)
    _reorderBackward.output.toTensor[Float].set(this.dense)

    // TODO delete this
    _memoryData = native

  }

  def createNative(): Unit = {
    native = DnnTensor[Float](dense.size())
  }

  def zero(): Unit = {
    dense.zero()
    native.zero()
  }

  def copy(t: Tensor[Float]): Unit = {
    dense.copy(t)
  }

  def size(): Array[Int] = {
    dense.size()
  }

  def size(index: Int): Int = {
    dense.size(index)
  }

  def release(): Unit = {
    if (native != null) {
      native.release()
    }
  }
}
