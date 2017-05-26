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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.log4j.Logger

import scala.reflect.ClassTag

object ShareGradInput {
  val logger = Logger.getLogger(getClass)

  def shareConvolution[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val fInputCache = Tensor[T](1)
    val fGradInputCache = Tensor[T](1)
    shareConvolution(model, fInputCache, fGradInputCache)
    model
  }

  def shareConvolution[T: ClassTag](
        model: Module[T],
        fInputCache: Tensor[T],
        fGradInputCache: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    model match {
      case container: Container[Activity, Activity, T] =>
        var i = 0
        while (i < container.modules.length) {
          val m = container.modules(i)
          if (m.isInstanceOf[SpatialConvolution[T]]) {
            val curModel = if (!m.isInstanceOf[SpatialShareConvolution[T]]) {
              SpatialShareConvolution(
                m.asInstanceOf[SpatialConvolution[T]])
            } else {
              m.asInstanceOf[SpatialShareConvolution[T]]
            }
            curModel.fInput.set(fInputCache)
            curModel.fGradInput.set(fGradInputCache)
            container.modules(i) = curModel
          } else {
            shareConvolution(m, fInputCache, fGradInputCache)
          }
          i += 1
        }
      case _ => Unit
    }
  }

  def shareGradInput[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val gradInputCache1 = Tensor[T](1)
    val gradInputCache2 = Tensor[T](1)
    shareGradInput(model, gradInputCache1, gradInputCache2)
    model
  }

  def shareGradInput[T: ClassTag](
      model: Module[T],
      gradInputCache1: Tensor[T],
      gradInputCache2: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    model match {
      case container: Sequential[T] =>
        var i = 0
        while (i < container.modules.length) {
          val m = container.modules(i)
          if (
             !m.isInstanceOf[SpatialConvolution[T]] &&
            !m.isInstanceOf[Container[Activity, Activity, T]] &&
            !m.isInstanceOf[SpatialAveragePooling[T]] &&
            !m.isInstanceOf[SpatialMaxPooling[T]] &&
            m.gradInput.isInstanceOf[Tensor[T]]) {
            if (i % 2 == 1) {
              m.gradInput.toTensor.set(gradInputCache1)
            } else {
              m.gradInput.toTensor.set(gradInputCache2)
            }
          }
          shareGradInput(m, gradInputCache1, gradInputCache2)
          i += 1
        }
      case container: Container[Activity, Activity, T] =>
        var i = 0
        while (i < container.modules.length) {
          val m = container.modules(i)
          shareGradInput(m, gradInputCache1, gradInputCache2)
          i += 1
        }
      case _ => Unit
    }

  }
}
