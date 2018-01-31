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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class MklDnnTensor[T: ClassTag](storage: Storage[T], storageOffset: Int,
  size: Array[Int], stride: Array[Int], nDimension: Int)(
  implicit ev: TensorNumeric[T])
  extends DenseTensor[T](storage, storageOffset, size, stride, nDimension) {

  def nativeStorage: AlignedStorage[T] = this.storage().asInstanceOf[AlignedStorage[T]]

  def sync(): Unit = {
    this.nativeStorage.sync()
  }


  override def getTensorType: TensorType = MklDnnType
}

object MklDnnTensor {
  MklDnn.isLoaded
  def apply[T: ClassTag](size: Array[Int])(implicit ev: TensorNumeric[T]): MklDnnTensor[T] = {
    val array = new Array[T](size.product)
    val storage = new AlignedStorage[T](array)
    val storageOffset = 0
    val stride = DenseTensor.size2Stride(size)
    val dim = size.length
    new MklDnnTensor[T](storage, storageOffset, size, stride, dim)
  }
}
