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

import com.intel.analytics.bigdl.mkl.Memory

import scala.reflect.ClassTag

class AlignedStorage[T: ClassTag](v: Array[T])
  extends ArrayStorage[T](v) {
  private val ELEMENT_SIZE: Int = {
    scala.reflect.classTag[T].toString() match {
      case "Float" => 4
      case "Double" => 8
      case _ => throw new IllegalArgumentException(errorMsg)
    }
  }
  private val CACHE_LINE_SIZE = 64
  private val DEBUG = false

  private val errorMsg = s"AlignedStorage only supports float"
  @transient private var _pointer = 0L
  @transient private var _conversion = false

  // initialize
  if (v != null) {
    allocate(v.length)
  }

  private def toFloat: Array[Float] = values.asInstanceOf[Array[Float]]
  private def toAlignedStorage(s: Storage[T]): AlignedStorage[T] = s.asInstanceOf[AlignedStorage[T]]
  private def allocate(capacity: Int): this.type = {
    require(capacity != 0, s"capacity should not be 0")
    require(_pointer == 0L, s"native storage should be freed first")
    val ptr = Memory.AlignedMalloc(capacity * ELEMENT_SIZE, CACHE_LINE_SIZE)
    _pointer = ptr
    require(_pointer != 0L, s"allocate native aligned memory failed")
    this
  }
  private def release(): this.type = {
    Memory.AlignedFree(native)
    _pointer = 0L
    this
  }

  def native: Long = _pointer
  def sync(): Unit = {
    require(native != 0, s"native storage has not been allocated")
    println("sync from heap array -> native storage")
    if (DEBUG) {
      for (ste <- Thread.currentThread().getStackTrace) {
        if (ste.toString.contains("com.intel.analytics.bigdl.")) {
          println("\t|----> " + ste)
        }
      }
    }
    Memory.CopyArray2Ptr(toFloat, 0, native, 0, length(), ELEMENT_SIZE)
  }
  def needConversion: Boolean = _conversion
  def setConversion(b: Boolean): Unit = { _conversion = b}

  override def apply(index: Int): T = array()(index)

  override def iterator: Iterator[T] = array().iterator

  override def array(): Array[T] = {
    if (needConversion) {
      println("convert from native storage -> heap array")
      if (DEBUG) {
        for (ste <- Thread.currentThread().getStackTrace) {
          if (ste.toString.contains("com.intel.analytics.bigdl.")) {
            println("\t|----> " + ste)
          }
        }
      }
      Memory.CopyPtr2Array(native, 0, toFloat, 0, length(), ELEMENT_SIZE)
      setConversion(false)
    }

    values
  }

  override def copy(source: Storage[T], offset: Int, sourceOffset: Int,
    length: Int): this.type = {
    source match {
      case s: ArrayStorage[T] =>
        System.arraycopy(s.array(), sourceOffset, array(), offset, length)
      case s: AlignedStorage[T] =>
        Memory.CopyPtr2Array(toAlignedStorage(source).native, sourceOffset, toFloat, offset, length,
          ELEMENT_SIZE)
      case s: Storage[T] =>
        var i = 0
        while (i < length) {
          this.values(i + offset) = s(i + sourceOffset)
          i += 1
        }
    }
    this
  }

  override def resize(size: Long): this.type = {
    this.values = new Array[T](size.toInt)
    this.release().allocate(size.toInt)
  }

  override def set(other: Storage[T]): this.type = {
    require(other.length() == this.length())
    this.values = other.array
    this
  }
}
