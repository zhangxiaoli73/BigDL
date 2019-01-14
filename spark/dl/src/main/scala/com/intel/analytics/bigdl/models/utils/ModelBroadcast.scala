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

package com.intel.analytics.bigdl.models.utils

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import java.util.UUID

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Container, Graph, StaticGraph}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, MklDnnContainer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn}
import com.intel.analytics.bigdl.utils.Util._
import com.intel.analytics.bigdl.utils.intermediate.{IRGraph, ReflectionUtils}
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * ModelBroadcast is used to broadcast model
 */
trait ModelBroadcast[T] extends Serializable {
  private val _uuid = UUID.randomUUID().toString

  /**
   * Broadcast the model
   * @param sc    SparkContext
   * @param model model to broadcast
   * @return this
   */
  def broadcast(sc: SparkContext, model: Module[T]): this.type

  /**
   * Get the broadcast model on worker
   *
   * @param initGradient If create a tensor for gradient when fetch the model. Please note that
   *                     the gradient is not needed in model inference
   * @return model
   */
  def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T]

  def uuid(): String = _uuid
}

object ModelBroadcast {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    if (System.getProperty("bigdl.ModelBroadcastFactory") != null) {
      val cls = Class.forName(System.getProperty("bigdl.ModelBroadcastFactory"))
      cls.getConstructors()(0).newInstance().asInstanceOf[ModelBroadcastFactory].create()
    } else {
      new DefaultModelBroadcastFactory().create()
    }
  }
}

/**
 * ModelBroadcast is used to broadcast model.
 *
 * Note: If you want to use this to broadcast training model, please use value(true) to get
 * the model. And before broadcasting please make sure the model's parameter is compacted.
 *
 * @tparam T data type
 * @param applyProtoBuffer it will use proto buffer serialization for broadcasting if set true
 */
private[bigdl] class ModelBroadcastImp[T: ClassTag](applyProtoBuffer: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {

  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _


  private def allSuperClass(model: Object, name: String) : Boolean = {
    var cls = model.getClass().asInstanceOf[Class[_]]
    var contains = false
    while (cls != null) {
      println(cls.getName())
      if (cls.getName == name) contains = true
      cls = cls.getSuperclass().asInstanceOf[Class[_]]
    }
    contains
  }
  /**
   * convert model to ir graph and build
   * @param model
   * @return
   */
  private def convertion(model: Module[T]): Module[T] = {
    println("222222222222222222")
    val name = "com.intel.analytics.zoo.models.common.ZooModel"
    val contains = allSuperClass(model, name)
    val tmpModel = if (contains) {
      model.asInstanceOf[Container[Activity, Activity, T]].modules(0)
    } else {
      model
    }
//    val tmpModel = if (model.isInstanceOf[Container[Activity, Activity, T]]) {
//      val con = model.asInstanceOf[Container[Activity, Activity, T]]
//      if (con.modules.length == 1) {
//       con.modules(0)
//      } else con
//    } else model

//    val cls = ReflectionUtils.findClass(name)
//    val tmpModel = if (cls != null) {
//      if (model.getClass.isAssignableFrom(cls)) {
//        model.asInstanceOf[Container[Activity, Activity, T]].modules(0)
//      } else model
//    } else {
//      model
//    }
   val m = if (!tmpModel.isInstanceOf[Graph[T]]) tmpModel.toGraph() else tmpModel
   if (!m.isInstanceOf[StaticGraph[T]]) return null
    if (Engine.getEngineType() == MklDnn) {
      m.asInstanceOf[StaticGraph[T]].toIRgraph().asInstanceOf[Module[T]]
    } else {
      m
    }
  }
  /**
   * set environment according to engine type and model type
   * @param model
   * @return
   */
  private def envSet(model: Module[T]): Module[T] = {
    println("dnngraph_dnngraph_dnngraph")
    val phase = if (model.isTraining()) TrainingPhase else InferencePhase
    model match {
      case container: MklDnnContainer => container.compile(phase)
      case graph: DnnGraph => graph.compile(phase)
      case _ =>
    }
    // todo: set core number = 1 for dnn engine type
    if (Engine.getEngineType() == MklDnn) {
      Engine.setNodeAndCore(Engine.nodeNumber(), 1)
    }
    model
  }

  /**
   * broadcast the model
   * first get and clear Const values from the model
   * then get and clear the weight and bias parameters from the model
   * finally broadcast Const values, the parameters and model(without parameters) separately
   * @param sc    SparkContext
   * @param model model to broadcast
   * @return this
   */
  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    CachedModels.deleteAll(uuid) // delete the models on driver

    val convertedModel = convertion(model)
    val modelNew = if (convertedModel != null) convertedModel else model

//     val modelNew = model

    if (applyProtoBuffer) {
      broadcastModel = sc.broadcast(ModelInfo(uuid, modelNew))
    } else {
      // broadcast Consts
      if (modelNew.isInstanceOf[Container[_, _, T]]) {
        val moduleConsts = getAndClearConsts(modelNew.asInstanceOf[Container[_, _, T]])
        // TODO: broadcast Const, model structure and weight in the same broadcast.
        broadcastConsts = sc.broadcast(moduleConsts)
      }
      // broadcast weight and model
      val weightsBias = getAndClearWeightBias(modelNew.parameters())
      broadcastModel = sc.broadcast(ModelInfo[T](uuid, modelNew))
      broadcastParameters = sc.broadcast(weightsBias)

      // For quantized model if we don't clone weightsBias, the original model will be released also
      // when we delete all models used in `ModelBroadcast`.
      // putWeightBias(SerializationUtils.clone(weightsBias), modelNew)
      // initGradWeightBias(weightsBias, modelNew)
    }
    this
  }

  /**
   * get the broadcast model
   * put the weight and bias back to the model
   *
   * @param initGradient If create a tensor for gradient when fetch the model. Please note that
   *                     the gradient is not needed in model inference
   * @return model
   */
  override def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T] = {
    CachedModels.deleteAll(uuid)
    val model = if (applyProtoBuffer) {
      val localModel = broadcastModel.value.model.clone(false)
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)

      if (initGradient) {
        initGradWeightBias(getWeightBias(localModel.parameters()), localModel)
      }
      localModel
    } else {
      val localModel = broadcastModel.value.model.cloneModule()
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)

      val parameters = if (shareWeight) {
        broadcastParameters.value
      } else {
        SerializationUtils.clone(broadcastParameters.value)
      }

      // share weight
      // tests
//      if (localModel.asInstanceOf[AbstractModule[Activity, Activity, T]].isInstanceOf[IRGraph[T]]) {
//        localModel.asInstanceOf[IRGraph[T]].build()
//      }

      // val p1 = localModel.getParameters()._1.clone()
       putWeightBias(parameters, localModel)
      // val p2 = localModel.getParameters()
      // share Consts
      if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
        putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
      }
      // init gradient
      if (initGradient) {
        initGradWeightBias(broadcastParameters.value, localModel)
      }
      localModel
    }

//    val convertedModel = convertion(model)
//    val modelNew = if (convertedModel != null) convertedModel else model
    envSet(model)
  }

  private def getWeightBias(parameters: (Array[Tensor[T]], Array[Tensor[T]]))
  : Array[Tensor[T]] = {
    if (parameters._1.length != 0) {
      var i = 0
      val weightsBias = new Array[Tensor[T]](parameters._1.length)
      val isQuantized = parameters._1.exists(_.getTensorType == QuantizedType)
      val (isCompacted, storage) = if (!isQuantized) {
        val storage = Storage(parameters._1(0).storage.array())
        (parameters._1.map(_.nElement()).sum == storage.length(), storage)
      } else {
        (false, null)
      }

      // get weight and bias
      while (i < parameters._1.length) {
        if (parameters._1(i) != null) {
          val wb = parameters._1(i)
          wb.getTensorType match {
            case QuantizedType =>
              val quantTensor = wb.asInstanceOf[QuantizedTensor[T]]
              weightsBias(i) = QuantizedTensor[T](quantTensor.getStorage, quantTensor.maxOfRow,
                quantTensor.minOfRow, quantTensor.sumOfRow, quantTensor.size(), quantTensor.params)
            case _ =>
              weightsBias(i) = if (isCompacted) {
                Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
              } else {
                Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
              }
          }
          i += 1
        }
      }
      weightsBias
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }
}

private[bigdl] class ModelInfo[T: ClassTag](val uuid: String, @transient var model: Module[T])(
  implicit ev: TensorNumeric[T]) extends Serializable {
  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    val cloned = model.cloneModule()
    out.writeObject(cloned)
    CachedModels.add(uuid, cloned)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    model = in.readObject().asInstanceOf[Module[T]]
    CachedModels.add(uuid, model)
  }
}

private[bigdl] object ModelInfo {
  def apply[T: ClassTag](uuid: String, model: Module[T])(
    implicit ev: TensorNumeric[T]): ModelInfo[T] = new ModelInfo[T](uuid, model)
}

private[bigdl] object CachedModels {
  import java.util.concurrent.ConcurrentHashMap

  import scala.collection._
  import scala.collection.convert.decorateAsScala._
  import scala.language.existentials

  type Modles = ArrayBuffer[Module[_]]

  private val cachedModels: concurrent.Map[String, Modles] =
    new ConcurrentHashMap[String, Modles]().asScala

  def add[T: ClassTag](uuid: String, model: Module[T])( implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val models = cachedModels.get(uuid) match {
        case Some(values) => values += model.asInstanceOf[Module[_]]
        case _ => ArrayBuffer(model.asInstanceOf[Module[_]])
      }
      cachedModels.put(uuid, models.asInstanceOf[Modles])
    }

  def deleteAll[T: ClassTag](currentKey: String)(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (key <- keys) {
        if (key != currentKey) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
    }

  def deleteKey[T: ClassTag](key: String)(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (k <- keys) {
        if (k == key) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
  }
}
