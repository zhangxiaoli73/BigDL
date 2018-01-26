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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor, UnsafeDirectByteBuffer}

import scala.reflect.ClassTag

class MemoryPrimitive[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Serializable {
  class Primitive extends Serializable {
    @transient var aligned: UnsafeDirectByteBuffer = _
    @transient var primitive: Long = 0L
  }

  class User extends Primitive {
    @transient var handle: Long = 0L
    val tensor: Tensor[T] = Tensor[T]()
  }

  class Internal extends Primitive

  val user: User = new User
  val internal: Internal = new Internal

  @transient var reorder: Long = 0L // reorder operation

  // TODO maybe it's a big tensor, which has been got handle from other layers.
  private def setUserHandle(userPrimitive: User): Unit = {
    val tensor = userPrimitive.tensor
    val primitive = userPrimitive.primitive

    val data = tensor.storage().array().asInstanceOf[Array[Float]]
    val offset = tensor.storageOffset() - 1

    require(userPrimitive.handle == 0L, s"You should release this handle first")
    userPrimitive.handle = MklDnn.MemorySetDataHandle(primitive, data, offset)

    // if the user storage is not cache aligned, we should reset the data handle
    if (!UnsafeDirectByteBuffer.isCacheAligned(userPrimitive.handle + offset)) {
      val handle = userPrimitive.handle
      val length = tensor.nElement()
      val buffer = new UnsafeDirectByteBuffer(length * 4, UnsafeDirectByteBuffer.CACHE_LINE_SIZE)
      MklDnn.MemorySetDataHandleWithBuffer(primitive, handle, offset, length,
        buffer.asFloatBuffer(), buffer.position)

      userPrimitive.aligned = buffer
    }
  }

  private def releaseUserHandle(userPrimitive: User): Unit = {
    val handle = userPrimitive.handle
    // we only need release the array
    if (handle != 0) {
      val data = userPrimitive.tensor.storage().array().asInstanceOf[Array[Float]]
      MklDnn.MemoryReleaseDataHandle(data, handle)
      userPrimitive.handle = 0L
    }
    userPrimitive.aligned = null
  }

  def tensor(t: Tensor[T]): Unit = {
    user.tensor.set(t)
  }

  def setHandle(): Unit = {
    require(ev.getType() == FloatType, s"only support float tensor currently")

    setUserHandle(user)

    if (internal.primitive != 0L) {
      val length = user.tensor.nElement()
      if (internal.aligned == null ||
        internal.aligned.asFloatBuffer().capacity() != user.tensor.nElement()) {
        internal.aligned = new UnsafeDirectByteBuffer(length * 4,
          UnsafeDirectByteBuffer.CACHE_LINE_SIZE)
      }

      user.tensor.getTensorNumeric().getType() match {
        case FloatType =>
          MklDnn.MemorySetDataHandleWithBuffer(internal.primitive,
            0, 0, length,
            internal.aligned.buffer.asFloatBuffer(),
            internal.aligned.position)
        case _ => throw new UnsupportedOperationException
      }
    }
  }

  def releaseHandle(isCopy: Boolean = false): Unit = {
    releaseUserHandle(user)

    if (isCopy && user.aligned != null) {
      MklDnn.copyFloatBuffer2Array(user.aligned.buffer.asFloatBuffer(), user.aligned.position,
        user.tensor.storage().array().asInstanceOf[Array[Float]],
        user.tensor.storageOffset() - 1,
        user.tensor.nElement())
    }

    // we do not reset the aligned buffer, because it costs too much every iteration.
    // by default, we think the size of buffer should be the same. If it's not the same,
    // we remalloc a new one, whose process is in `setUserHandle`
  }

  def workPrim(): Long = {
    if (internal.primitive != 0L) {
      internal.primitive
    } else {
      user.primitive
    }
  }

  private def init1(primDesc: Long): Long = {
    MklDnn.PrimitiveCreate0(primDesc)
  }

  private def init4(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Long = {
    // TODO refactor for linear
    val (dim, size) = if (tensor.dim() == 1 && (format == MklDnn.MemoryFormat.nc ||
      format == MklDnn.MemoryFormat.oi)) {
      (2, Array(1) ++ tensor.size())
    } else if (tensor.dim() == 2 && (format == MklDnn.MemoryFormat.oihw)) {
      (4, tensor.size() ++ Array(1, 1))
    } else {
      (tensor.dim(), tensor.size())
    }

    val desc = MklDnn.MemoryDescInit(dim, size, dataType, format)
    val primDesc = MklDnn.MemoryPrimitiveDescCreate(desc, engine)
    val primitive = MklDnn.PrimitiveCreate0(primDesc)

    MklDnn.PrimitiveDescDestroy(primDesc)
    primitive
  }

  def initUser(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Unit = {
    val primDesc = tensor.getPrimitiveDesc()
    user.primitive = if (primDesc != 0L) { // if the tensor comes from mkldnn layer
      init1(primDesc)
    } else {
      init4(tensor, dataType, format, engine)
    }
  }

  def initUser(tensor: Tensor[T], layerPrimDesc: Long, queryType: Int, index: Int): Unit = {
    val primDesc = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    user.primitive = MklDnn.PrimitiveCreate0(primDesc)

    tensor.setPrimitiveDesc(primDesc)
  }

  def initInternal(layerPrimDesc: Long, queryType: Int): Unit = {
    val primDescFromLayer = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    val res = MklDnnOps.prepareReorder(user.primitive, primDescFromLayer, user_to_prim = true)
    internal.primitive = res._2
    reorder = res._1
  }
}
