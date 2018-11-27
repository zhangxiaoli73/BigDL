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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, T}

import scala.collection.mutable
import scala.reflect.ClassTag


abstract class ConvertBase[T, D] {
  /**
    * clone node relations
    * @param oldToNew node element maps from T to D
    */
  def cloneNode(oldToNew: mutable.HashMap[Node[T], Node[D]]): Unit = {
    oldToNew.keySet.toArray.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.contains(nextNodeAndEdge._1)) {
          oldToNew.get(node).get.add(oldToNew.get(nextNodeAndEdge._1).get, nextNodeAndEdge._2)
        }
      })
    })
  }

  def enableConvertLayer(layer: T) : Boolean

  def convertLayer(layer : T) : D

  def enableConvert(allNodes: Array[Node[T]]) : Boolean = {
    var convert = true
    allNodes.foreach(node => {
      if (!enableConvertLayer(node.element)) {
        convert = false
      }
    })
    // TODO : log false element name
    convert
  }

  def convert(allNodes: Array[Node[T]]): mutable.HashMap[Node[T], Node[D]] = {
    val oldToNew = new mutable.HashMap[Node[T], Node[D]]()
    allNodes.foreach(node => {
      oldToNew.put(node, new Node(convertLayer(node.element)))
    })
    cloneNode(oldToNew)
    oldToNew
  }
}

class IRToBlas[T: ClassTag] extends ConvertBase[IRElement[T], Module[T]]{

  override def enableConvertLayer(layer: IRElement[T]): Boolean = {
    val name = layer.getOp().name
    val className = "com.intel.analytics.bigdl.nn." + name.substring(2)
    val cls = ReflectUtils.classFound(className)
    if ( cls != null) true
    else false
  }

  override def convertLayer(layer : IRElement[T]) : Module[T] = {
    val name = layer.getOp().name
    val cls = Class.forName("com.intel.analytics.bigdl.nn." + name.substring(2))
    ReflectUtils.reflectFromIR(layer, cls)
  }
}

object IRToBlas {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): IRToBlas[T] = new IRToBlas
}

class BlasToIR[T: ClassTag] extends ConvertBase[Module[T], IRElement[T]]{

  // todo: some undefined IR operations can be presented by IRBlasModule
  override def enableConvertLayer(layer: Module[T]): Boolean = {
    val layerName = layer.getClass.getSimpleName
    val className = "com.intel.analytics.bigdl.utils.mkldnn.IR" + layerName
    val cls = ReflectUtils.classFound(className)
    if ( cls != null) return true
    if (layer.isInstanceOf[TensorModule[T]]) true
    else false
  }

  override def convertLayer(layer : Module[T]) : IRElement[T] = {
    val layerName = layer.getClass.getSimpleName
    val className = "com.intel.analytics.bigdl.utils.mkldnn.IR" + layerName
    val cls = ReflectUtils.classFound(className)
    if ( cls != null) {
      ReflectUtils.reflectToIR(layer, cls)
    } else if (layer.isInstanceOf[TensorModule[T]]) {
      val op = IRBlasModule[T](layer.asInstanceOf[TensorModule[T]])
      IRElement(layer.getName(), op)
    } else {
      throw new UnsupportedOperationException(s"can not convert $layer to IRelement ")
    }
  }
}

object BlasToIR {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): BlasToIR[T] = new BlasToIR
}
