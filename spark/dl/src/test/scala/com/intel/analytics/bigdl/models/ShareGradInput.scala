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
import com.intel.analytics.bigdl.models.resnet.ResNet.getClass
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.log4j.Logger

import scala.collection.mutable

object ShareGradInput {
  val logger = Logger.getLogger(getClass)

  def shareGradInput(model: Module[Double]): Unit = {
    logger.info("Share gradients in ResNet")

    def sharingKey(m: Module[Double]) = m.getClass.getName

    val cache = mutable.Map[Any, Storage[Double]]()
    val packageName: String = model.getName().stripSuffix("Sequential")
    cache.put("fInput", Storage(Array(1.0)))
    cache.put("fGradInput", Storage(Array(1.0)))

    var index = 0

    def matchModels(model: Module[Double]): Unit = {
      model match {
        case container: Container[Activity, Activity, Double] =>
          container.modules.foreach(m => {
            if (m.gradInput.isInstanceOf[Tensor[_]] &&
               !m.getClass.getName.endsWith("ConcatTable")
//            && !m.getClass.getName.endsWith("Dropout")
//            && !m.getClass.getName.endsWith("LogSoftMax")
//            && !m.getClass.getName.endsWith("SpatialCrossMapLRN")
//            && !m.getClass.getName.endsWith("SpatialAveragePooling")
//            && !m.getClass.getName.endsWith("CAddTable")
//            && !m.getClass.getName.endsWith("SpatialBatchNormalization")
//            && !m.getClass.getName.endsWith("BatchNormalization")
//            && !m.getClass.getName.endsWith("Linear")
//            && !m.getClass.getName.endsWith("SpatialMaxPooling")
//            && !m.getClass.getName.endsWith("View")
//            && !m.getClass.getName.endsWith("ReLU")
//            && !m.getClass.getName.endsWith("Concat")
            && !m.getClass.getName.endsWith("SpatialConvolution")
//              && !m.getClass.getName.endsWith("SpatialShareConvolution")
//              && !m.getClass.getName.endsWith("SpatialShareConvolution")
//              && !m.getClass.getName.endsWith("SpatialZeroPadding")
            ) {
              val key = sharingKey(m)
              if (!cache.contains(key)) {
                cache.put(key, Storage(Array(1.0)))
              }

              m.gradInput = Tensor(cache.get(key).get, 1, Array(0))
            }
            matchModels(m)
          })
        case concatTable if (concatTable.isInstanceOf[ConcatTable[Double]]) =>
          if (!cache.contains(index % 2)) {
            cache.put(index % 2, Storage(Array(1.0)))
          }
          concatTable.gradInput = Tensor[Double](cache.get(index % 2).get, 1, Array(0))
          index = index + 1
        case spatialShareConvolution
          if (spatialShareConvolution.isInstanceOf[SpatialShareConvolution[Double]]) =>
          val curModel = spatialShareConvolution.asInstanceOf[SpatialShareConvolution[Double]]
          curModel.fInput = Tensor[Double](cache.get("fInput").get)
          curModel.fGradInput = Tensor[Double](cache.get("fGradInput").get)
        case _ => Unit
      }
    }

    matchModels(model)
  }

  def modelInit(model: Module[Double]): Unit = {
    logger.info("Initialize ResNet")
    def initModules(model: Module[Double]): Unit = {
      model match {
        case container: Container[Activity, Activity, Double]
        => container.modules.foreach(m => initModules(m))
        case spatialShareConvolution
          if (spatialShareConvolution.isInstanceOf[SpatialShareConvolution[Double]]) =>
          val curModel = spatialShareConvolution.asInstanceOf[SpatialShareConvolution[Double]]
          val n: Double = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, Math.sqrt(2.0f / n)).toDouble)
          curModel.bias.apply1(_ => 0.0f)
        case spatialConvolution
          if (spatialConvolution.isInstanceOf[SpatialConvolution[Double]]) =>
          val curModel = spatialConvolution.asInstanceOf[SpatialConvolution[Double]]
          val n: Double = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, Math.sqrt(2.0f / n)).toDouble)
          curModel.bias.apply1(_ => 0.0f)
        case spatialBatchNormalization
          if (spatialBatchNormalization.isInstanceOf[SpatialBatchNormalization[Double]]) =>
          val curModel = spatialBatchNormalization.asInstanceOf[SpatialBatchNormalization[Double]]
          curModel.weight.apply1(_ => 1.0f)
          curModel.bias.apply1(_ => 0.0f)
        case linear if (linear.isInstanceOf[Linear[Double]]) =>
          linear.asInstanceOf[Linear[Double]].bias.apply1(_ => 0.0f)
        case _ => Unit
      }
    }
    initModules(model)
  }
}
