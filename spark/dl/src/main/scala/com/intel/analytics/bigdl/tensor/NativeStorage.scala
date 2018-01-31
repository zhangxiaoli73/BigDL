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

import com.intel.analytics.bigdl.mkl.MklDnn

import scala.reflect.ClassTag

class NativeStorage[T: ClassTag](val _capacity: Int) extends Storage[T] {
  val errorMsg = s"AlignedStorage only supports float"
  val CACHE_LINE_SIZE = 64

  private var _ptr: Long = allocate(_capacity)

  private def allocate(capacity: Int): Long = {
    val size = scala.reflect.classTag[T].toString() match {
      case "Int" => 4
      case "Float" => 4
      case "Double" => 8
    }
    synchronized {
      NativeStorage.allocatedSize += _capacity
    }

    MklDnn.MemoryAlignedMalloc(_capacity * 4, CACHE_LINE_SIZE)
  }

  def release(): Unit = {
    synchronized {
      NativeStorage.allocatedSize -= _capacity
    }
    MklDnn.MemoryAlignedFree(_ptr)
  }

  override def finalize(): Unit = {
    try {
      release()
    } finally {
      super.finalize()
    }
  }

  override def length(): Int = _capacity

  override def apply(index: Int): T = throw new UnsupportedOperationException

  override def update(index: Int, value: T): Unit = throw new UnsupportedOperationException

  override def copy(source: Storage[T], offset: Int, sourceOffset: Int, length: Int): this.type = {
    throw new UnsupportedOperationException
  }

  override def fill(value: T, offset: Int, length: Int): this.type = {
    throw new UnsupportedOperationException
  }

  override def resize(size: Long): this.type = {
    throw new UnsupportedOperationException
  }

  override def array(): Array[T] = throw new UnsupportedOperationException

  override def set(other: Storage[T]): this.type = throw new UnsupportedOperationException

  override def iterator: Iterator[T] = throw new UnsupportedOperationException
}

object NativeStorage {
  private var allocatedSize = 0
  def apply[T: ClassTag](_capacity: Int): NativeStorage[T] = new NativeStorage[T](_capacity)
}
