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

package com.intel.analytics.bigdl.tensor

import java.nio._

import com.intel.analytics.bigdl.mkl.MklDnn

class UnsafeDirectByteBuffer(capacity: Int, align: Int) {
  private var _buffer: ByteBuffer = _ // allocate(capacity, align)
  private var _position: Int = 0
  private var _ptr: Long = allocateWithPtr(capacity, align)

  def buffer: ByteBuffer = _buffer

  def position: Int = _position

  def ptr: Long = _ptr

  def allocateWithPtr(capacity: Int, align: Int): Long = {
    MklDnn.MemoryAlignedMalloc(capacity, align)
  }

  def releaseWithPtr(): Unit = {
    if (_ptr != 0) {
      MklDnn.MemoryAlignedFree(_ptr)
    }
  }

  // FIXME capacity should be Long ??
  def allocate(capacity: Int, align: Int): ByteBuffer = {
    println("=" * 80)
    println(s"allocate $capacity bytes")
    println("=" * 80)
    import UnsafeDirectByteBuffer._

    if (Integer.bitCount(align) != 1) {
      throw new IllegalArgumentException(s"Alignment must be a power of 2")
    }

    if (capacity <= 0) {
      throw new IllegalArgumentException(s"Capacity must be more than 0")
    }

    val buffer = ByteBuffer.allocateDirect(capacity * align)

    val slice = if (isCacheAligned(buffer)) {
      buffer.limit(capacity)
      buffer.slice().order(ByteOrder.nativeOrder())
    } else {
      _position = (align - (address(buffer) & (align - 1))).toInt
      val limit = position + capacity

      buffer.position(position)
      buffer.limit(limit)
      buffer.mark()
      buffer.slice().order(ByteOrder.nativeOrder())
    }

    slice
  }

  def asFloatBuffer(): FloatBuffer = buffer.asFloatBuffer()

  def asDoubleBuffer(): DoubleBuffer = buffer.asDoubleBuffer()

  def asIntBuffer(): IntBuffer = buffer.asIntBuffer()

  def release(): Unit = {
    _buffer = null
    _position = 0
  }
}

object UnsafeDirectByteBuffer {
  val CACHE_LINE_SIZE = 64

  def offset: Long = {
    unsafeInstance.objectFieldOffset(classOf[Buffer].getDeclaredField("address"))
  }

  def unsafeInstance: sun.misc.Unsafe = {
    val field = classOf[sun.misc.Unsafe].getDeclaredField("theUnsafe")
    field.setAccessible(true)
    field.get(null).asInstanceOf[sun.misc.Unsafe]
  }

  def address(buffer: ByteBuffer): Long = {
    unsafeInstance.getLong(buffer, offset)
  }

  def isCacheAligned(buffer: ByteBuffer): Boolean = {
    (address(buffer) & (CACHE_LINE_SIZE - 1)) == 0
  }

  def isCacheAligned(address: Long): Boolean = {
    (address & (CACHE_LINE_SIZE - 1)) == 0
  }
}

