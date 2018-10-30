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

import com.intel.analytics.bigdl.serialization.Bigdl.AttrValue
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.sql.execution.streaming
import org.apache.spark.sql.execution.streaming.state
import org.tensorflow.framework.NodeDef

class IRElement(
  private val name: String,
  private val op_type: String,
  private val attr_map: Map[String, Any]) {


  private var inputs : Seq[String] = _
  private var attrMap : Map[String, Any] = attr_map

  // Get the element name
  final def getName() : String = {
    this.name
  }

  final def getOp() : String = {
    this.op_type
  }

  final def getAttrMap() : Map[String, Any] = {
    attrMap
  }

//  final def setAttr[T](key: String, value: Any): Unit = {
//    if (attrMap.contains(key)) {
//    // todo: add warning
//    } else {
//      attrMap(key) = value
//    }
//  }

  final def getAttr[T](key: String) : Option[T] = {
    attrMap.get(key).map(_.asInstanceOf[T])
  }

  // option methos
  final def setinput(in: Seq[String]) : Unit = {
    inputs = in
  }

  final def getinput() : Seq[String] = {
    inputs
  }

}

class TFElement(
  private val name: String,
  private val op_type: String,
  private val attr_map: Map[String, Any],
  private val tf_layer: NodeDef)
  extends IRElement(name, op_type, attr_map) {

  private var layer : NodeDef = tf_layer

  def getLayer() : NodeDef = {
    layer
  }
}

object IRElement {
  def apply(name: String, op_type: String, attr_map: Map[String, Any]): IRElement =
    new IRElement(name, op_type, attr_map)

  def apply(name: String, op_type: String, attr_map: Map[String, Any], tf_layer: NodeDef)
  : IRElement =
    new TFElement(name, op_type, attr_map, tf_layer)
}