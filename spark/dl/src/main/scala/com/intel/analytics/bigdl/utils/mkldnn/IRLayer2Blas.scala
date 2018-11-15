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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.nn.tf.MaxPool
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, DataType}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import com.intel.analytics.bigdl.utils.caffe.CaffeConversionException
import com.intel.analytics.bigdl.utils.serializer.DeserializeContext
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer.getCostructorMirror

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.{ClassTag, ManifestFactory}
import scala.reflect.runtime.universe


class IRLayer2Blas[T: ClassTag](implicit ev: TensorNumeric[T]) {
  def enableConvert(layer: IRElement[T]) : Boolean = {
    val name = layer.getOp().name
    // todo: not sure
    val cls = Class.forName("com.intel.analytics.bigdl.nn." + name.substring(2))
    if ( cls != null) true
    else false
  }

  def convertIRLayer(layer : IRElement[T]) : Module[T] = {
    val name = layer.getOp().name
    val cls = Class.forName("com.intel.analytics.bigdl.nn." + name.substring(2))
    ReflectionUtils.reflectFromIR(layer, cls)
  }
}

object IRLayer2Blas {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRLayer2Blas[T] = new IRLayer2Blas
}