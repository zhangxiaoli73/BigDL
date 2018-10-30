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

import org.apache.commons.lang.exception.ExceptionUtils
import java.util
import java.util.List

import breeze.linalg.reverse
import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}
import org.tensorflow.framework.NodeDef

import scala.collection.mutable
import scala.reflect.ClassTag


abstract class IRConverter[T: ClassTag](defGraph: IRGraph[T])
                                       (implicit ev: TensorNumeric[T]) {

  def mapInit(): Unit = {}

  def toGraph(): Graph[T]

  def enableConvert(): Boolean
}