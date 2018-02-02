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

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

class MemoryPrimitive[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Serializable {
  class Primitive extends Serializable {
    @transient var aligned: UnsafeDirectByteBuffer = _
    @transient var primitive: Long = 0L
    @transient val tensor: MklDnnTensor[T] = MklDnnTensor[T](Array(1))
  }

  class User extends Primitive {
    @transient var handle: Long = 0L
  }

  class Internal extends Primitive

  val user: User = new User
  val internal: Internal = new Internal

  @transient var reorder: Long = 0L // reorder operation

  // TODO maybe it's a big tensor, which has been got handle from other layers.
  private def setUserHandle(userPrimitive: User): Unit = {
    val tensor = userPrimitive.tensor
    val primitive = userPrimitive.primitive

    require(userPrimitive.handle == 0L, s"You should release this handle first")
    userPrimitive.handle = Memory.SetDataHandle(primitive, tensor.nativeStorage, 0)
  }

  def tensor(t: Tensor[T]): Unit = {
    user.tensor.set(t)
  }

  def setHandle(tensor: MklDnnTensor[T], needUpdate: Boolean = false): Unit = {
    require(ev.getType() == FloatType, s"only support float tensor currently")
    if (!tensor.storage().asInstanceOf[AlignedStorage[T]].needConversion) {
      if (!needUpdate) {
        tensor.syncFromHeap()
      } else {
        tensor.storage().asInstanceOf[AlignedStorage[T]].setConversion(needUpdate)
      }
    }
    Memory.SetDataHandle(user.primitive, tensor.nativeStorage, 0)

    if (internal.primitive != 0L) {
      internal.tensor.resize(tensor.size())

      tensor.getTensorNumeric().getType() match {
        case FloatType =>
          Memory.SetDataHandle(internal.primitive, internal.tensor.nativeStorage, 0)
        case _ => throw new UnsupportedOperationException
      }
    }
  }

  def releaseHandle(isCopy: Boolean = false): Unit = {
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
