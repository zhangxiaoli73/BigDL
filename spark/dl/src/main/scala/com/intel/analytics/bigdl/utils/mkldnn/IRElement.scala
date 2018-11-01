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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.serialization.Bigdl.AttrValue
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.sql.execution.streaming
import org.apache.spark.sql.execution.streaming.state
import org.tensorflow.framework.NodeDef

sealed class IROperate

object IROperate {

  case object SpatialMaxPooling extends IROperate {
    var data_format : String = ""
    var strides : Seq[Int] = _
    var ksize : Seq[Int] = _
  }

  case object SpatialAvePooling extends IROperate {
    var data_format : String = ""
    var strides : Seq[Int] = _
    var ksize : Seq[Int] = _
    var countIncludePad : Boolean = true
  }

  case object SpatialConv extends IROperate {

  }

  case object SpatialBatchNorm extends IROperate {

  }

  case object Identity extends IROperate {

  }

  case object DropOut extends IROperate {

  }

  case object ReLu extends IROperate {

  }

  case object Linear extends IROperate

  case object Squeeze extends IROperate

  case object LRN extends IROperate {
    var size: Int = 5
    var alpha: Double = 1.0
    var beta: Double = 0.75
    var k: Double = 1.0
    var format: DataFormat = DataFormat.NCHW
  }

  case object Input extends IROperate

  case object Output extends IROperate
}

private[bigdl] class IRElement(
  private val name: String,
  private val op_type: IROperate) {

  private var inputs : Seq[String] = _

  // Get the element name
  final def getName() : String = {
    this.name
  }

  final def getOp() : OperateType = {
    this.op_type
  }

  final def getAttrMap() : Map[String, Any] = {
    attr_map
  }

//  final def setAttr[T](key: String, value: Any): Unit = {
//    if (attrMap.contains(key)) {
//    // todo: add warning
//    } else {
//      attrMap(key) = value
//    }
//  }

  final def getAttr[T](key: String) : Option[T] = {
    attr_map.get(key).map(_.asInstanceOf[T])
  }

  // option methos
  final def setinput(in: Seq[String]) : Unit = {
    inputs = in
  }

  final def getinput() : Seq[String] = {
    inputs
  }

}

private[bigdl] class TFElement(
  private val name: String,
  private val op_type: IROperate,
  private val tf_layer: NodeDef)
  extends IRElement(name, op_type) {

  private var layer : NodeDef = tf_layer

  def getLayer() : NodeDef = {
    layer
  }
}

object IRElement {
  def apply(name: String, op_type: IROperate, attr_map: Map[String, Any]): IRElement =
    new IRElement(name, op_type)

  def apply(name: String, op_type: IROperate, attr_map: Map[String, Any], tf_layer: NodeDef)
  : IRElement =
    new TFElement(name, op_type, tf_layer)
}