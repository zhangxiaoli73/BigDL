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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer._

import scala.collection.mutable
import scala.reflect.{ClassTag, ManifestFactory}
import scala.reflect.runtime._

object ReflectionUtils {

  def getFiledNameAndValues(o: Object): mutable.HashMap[String, AnyRef] = {
    val c = o.getClass
    var fields = c.getDeclaredFields
    val superFields = c.getSuperclass.getDeclaredFields
    fields = fields ++ superFields

    val values = new mutable.HashMap[String, AnyRef]()
    var i = 0
    while (i < fields.length) {
      val field = fields(i)
      val name = field.getName
      field.setAccessible(true)
      values(name) = field.get(o)
      i += 1
    }
    values
  }

  // create layer2 object form layer1
  def reflection(layer1: Object, layer2: Class[_]) : Object = {
    val nameAndValues = getFiledNameAndValues(layer1)
    val constructorMirror = getCostructorMirror(layer2)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams.map(_.size).sum)

    var i = 0
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (ptype <:< universe.typeOf[ClassTag[_]]||
          ptype.typeSymbol == universe.typeOf[ClassTag[_]].typeSymbol) {
          // todo: check
          args(i) = ManifestFactory.Float
        } else if (ptype <:< universe.typeOf[TensorNumeric[_]]
          || ptype.typeSymbol == universe.typeOf[TensorNumeric[_]].typeSymbol) {
          // todo: check
          args(i) = TensorNumeric.NumericFloat
        } else {
          val value = nameAndValues.get(name).getOrElse(null)
          args(i) = value
        }
        i += 1
      })
    })
    constructorMirror.apply(args : _*).asInstanceOf[Object]
  }
}
