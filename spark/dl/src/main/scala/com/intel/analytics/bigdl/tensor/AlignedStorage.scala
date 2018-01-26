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

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.mkl.MklDnn

import scala.reflect.ClassTag

class AlignedStorage[T: ClassTag](size: Int) extends Storage[T] {
  require(size > 0, s"[AlignedStorage] size should be more than 0")

  val errorMsg = s"AlignedStorage only supports float"

  private var _buffer: ByteBuffer = _
  private val dataType = scala.reflect.classTag[T]

  // init
  resize(size)

  def buffer: ByteBuffer = _buffer

  def this(data: Array[T]) = {
    this(data.length)

    dataType.toString() match {
      case "Float" =>
        _buffer.asFloatBuffer().put(data.asInstanceOf[Array[Float]])
      case "Int" =>
        _buffer.asIntBuffer().put(data.asInstanceOf[Array[Int]])
      case "Double" =>
        _buffer.asDoubleBuffer().put(data.asInstanceOf[Array[Double]])
    }
  }

  override def length(): Int = {
    buffer.capacity()
  }

  override def apply(index: Int): T = {
    dataType.toString match {
      case "Float" => buffer.asFloatBuffer().get(index).asInstanceOf[T]
      case _ => throw new UnsupportedOperationException(errorMsg)
    }
  }

  override def update(index: Int, value: T): Unit = {
    value match {
      case v: Float => buffer.asFloatBuffer().put(index, v)
      case _ => throw new UnsupportedOperationException(errorMsg)
    }
  }

  override def copy(source: Storage[T], offset: Int, sourceOffset: Int, length: Int): this.type = {
    source match {
      case s: ArrayStorage[T] =>
        dataType.toString() match {
          case "Float" =>
            MklDnn.copyArray2FloatBuffer(buffer.asFloatBuffer(), offset,
              s.array().asInstanceOf[Array[Float]], sourceOffset, length)
          case _ => throw new UnsupportedOperationException(errorMsg)
        }
      case as: AlignedStorage[T] =>
        if (length == as.buffer.capacity() && sourceOffset == 0) {
          this.buffer.put(as.buffer)
        } else {
          // TODO refactor && performance.
          dataType.toString() match {
            case "Float" =>
              val src = as.buffer.asFloatBuffer()
              val dst = buffer.asFloatBuffer()
              var i = 0
              while (i < length) {
                src.put(offset + i, dst.get(sourceOffset + i))
                i += 1
              }
            case _ => throw new UnsupportedOperationException(errorMsg)
          }
        }
      case _ =>
        val errorMsg = s"AlignedStorage doesn't support this type of Storage"
        throw new UnsupportedOperationException(errorMsg)
    }
    this
  }

  override def fill(value: T, offset: Int, length: Int): this.type = {
    value match {
      case v: Float => MklDnn.fillFloatBuffer(buffer.asFloatBuffer(), offset, v, length)
      case _ => throw new UnsupportedOperationException(errorMsg)
    }

    this
  }

  override def resize(size: Long): this.type = {
    if (size != buffer.capacity()) {
      val capacity = dataType.toString() match {
        case "Float" => 4
        case "Int" => 4
        case "Double" => 8
        case _ => throw new UnsupportedOperationException(errorMsg)
      }
      val directByteBuffer = new UnsafeDirectByteBuffer(size.toInt * capacity, 64)
      _buffer = directByteBuffer.buffer
    }
    this
  }

  override def array(): Array[T] = {
    val errorMsg = s"AlignedStorage doesn't support array method"
    throw new UnsupportedOperationException(errorMsg)
  }

  override def set(other: Storage[T]): this.type = {
    require(other.length() == this.length())
    require(other.isInstanceOf[AlignedStorage[T]], s"Only support AlignedStorage now.")

    _buffer = other.asInstanceOf[AlignedStorage[T]].buffer
    this
  }

  override def iterator: Iterator[T] = {
    val errorMsg = s"AlignedStorage doesn't support iterator method"
    throw new UnsupportedOperationException(errorMsg)
  }
}
