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

package com.intel.analytics.bigdl.utils.mkldnn

import java.util

import com.intel.analytics.bigdl.nn.{DynamicGraph, Graph, StaticGraph, mkldnn}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.nn.tf.MaxPool
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import com.intel.analytics.bigdl.utils.caffe.CaffeConversionException
import com.intel.analytics.bigdl.utils.mkldnn.OperateType.{DropOut, SpatialConv}
import com.intel.analytics.bigdl.utils.tf.loaders.TensorflowOpsLoader

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class IRLayer2Blas[T: ClassTag](implicit ev: TensorNumeric[T]) {

  private val IR2BlasMap = new mutable.HashMap[OperateType, (IRElement) => Module[T]]

  mapInit()

  def enableConvert(layer: IRElement) : Boolean = {
    val layerType = layer.getOp()
    if (IR2BlasMap.contains(layerType)) true
    else false
  }

  def convertIRLayer(layer : IRElement) : Module[T] = {
    val layerType = layer.getOp()
    require(IR2BlasMap.contains(layerType), s"not support convert ${layerType} to dnn layer")
    IR2BlasMap(layerType)(layer)
  }

  private def mapInit(): Unit = {
    IR2BlasMap(OperateType.DropOut) = fromDropout
    IR2BlasMap(OperateType.Identity) = fromIdentity
    IR2BlasMap(OperateType.SpatialConv) = fromConv
    IR2BlasMap(OperateType.MaxPool) = fromMaxPooling
    // todo: not complete
  }

  private def fromConv(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromMaxPooling(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromSelectTable(node: IRElement) : Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromSpatialBatchNormalization(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromTranspose(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromContiguous(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromTemporalConvolution(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromLinear(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromDropout(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

  private def fromIdentity(node: IRElement): Module[T] =
    throw new UnsupportedOperationException("not implement")

}

object IRLayer2Blas {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRLayer2Blas[T] = new IRLayer2Blas
}