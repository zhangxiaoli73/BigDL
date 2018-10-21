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

package com.intel.analytics.bigdl.nn

import java.nio.ByteOrder

import scala.reflect.ClassTag
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.{AssignGrad, SwitchControlNode, SwitchOps}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node}
import com.intel.analytics.bigdl.utils.tf.Context
import com.intel.analytics.bigdl.utils.tf.loaders.TensorflowOpsLoader
import org.tensorflow.framework.NodeDef

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class ConvertHelper[T: ClassTag, D](generalGraph: GeneralGraph[T, D]) {

//  def convertToDnn(): Module[T] = {
//
//  }

//  def convertToStaticGraph(): Module[T] = {
//
//  }

  private val dnnSupportLayers = null

}
