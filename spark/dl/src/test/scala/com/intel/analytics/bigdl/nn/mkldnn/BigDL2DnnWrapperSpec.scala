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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{Squeeze, mkldnn}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.serialization.Bigdl.DataType
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import org.apache.spark.ml
import org.apache.spark.ml.attribute

import scala.reflect.{ClassTag, ManifestFactory}
import scala.reflect.runtime._
import scala.util.Random

class BigDL2DnnWrapperSpec extends BigDLSpecHelper {

  // from nhwc -> nchw
  def shapeToNCHW(shape: Array[Int]): Array[Int] = {
   Array(shape(0), shape(3), shape(1), shape(2))
  }

  def wrapSqueeze(inputShape: Array[Int]) : MklDnnContainer = {
    val model1 = mkldnn.Sequential()

    model1.add(mkldnn.Input(inputShape, Memory.Format.nhwc))
    val s = Squeeze[Float](Array(2, 3), true).
      asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]
    model1.add(BigDL2DnnWrapper(s, ""))

    model1
  }

  def wrapConv(inputShape: Array[Int], gradOutShape : Array[Int]) : MklDnnContainer = {
    import com.intel.analytics.bigdl.nn
    RNG.setSeed(100)

    val model1 = mkldnn.Sequential()
    model1.add(mkldnn.Input(inputShape, Memory.Format.nhwc).setName("input"))

    val s = nn.SpatialConvolution[Float](3, 32, 5, 5, 1, 1).
      asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]

    model1.add(BigDL2DnnWrapper(s, "").setName("wrapper"))
    model1.add(ReorderMemory(inputFormat = null, outputFormat = null,
      gradInputFormat = null,
      gradOutputFomat = HeapData(gradOutShape, Memory.Format.nhwc)).setName("test"))
    model1
  }

   "wrapper squeeze" should "be correct" in {
     val in = Tensor[Float](2, 1, 1, 3).rand()
     val inNHWC = in.transpose(2, 4).transpose(3, 4).contiguous().clone()

     val model = Squeeze[Float](Array(2, 3), true).
       asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]

     val wrapperModel = wrapSqueeze(inNHWC.size())
     wrapperModel.compile(Phase.InferencePhase)

     val outWrapper = wrapperModel.forward(inNHWC)
     val out = model.forward(in)

     outWrapper should be(out)
   }

  "wrapper conv" should "be correct" in {
    val nIn = 3
    val nOut = 32
    val kH = 5
    val kW = 5
    RNG.setSeed(100)
    val in = Tensor[Float](4, 3, 7, 7).apply1(_ => RNG.uniform(0.1, 1).toFloat)
    val inNHWC = in.transpose(2, 3).transpose(3, 4).contiguous().clone()

    val gradOutput = Tensor[Float](4, 32, 3, 3).apply1(_ => RNG.uniform(0.1, 1).toFloat)
    val gradOutputNHWC = gradOutput.transpose(2, 3).transpose(3, 4).contiguous().clone()


    RNG.setSeed(100)
    val model = nn.SpatialConvolution[Float](3, 32, 5, 5, 1, 1).
      asInstanceOf[AbstractModule[Tensor[_], Tensor[_], Float]]

    // wrapper
    val wrapperModel = wrapConv(inNHWC.size(), gradOutputNHWC.size())
    wrapperModel.compile(Phase.TrainingPhase)

    val wrapperOut = wrapperModel.forward(inNHWC)
    val out = model.forward(in)

    wrapperOut.equals(out) should be(true)

    // for backward
    val grad = model.backward(in, gradOutput)
    val wrapperGrad = wrapperModel.backward(inNHWC, gradOutputNHWC)
    val gradNHWC = grad.transpose(2, 3).transpose(3, 4).contiguous().clone()

    gradNHWC should be(wrapperGrad)
  }

  "test reflection" should "be right" in {
    val cls = Class.forName("com.intel.analytics.bigdl.nn.SpatialConvolution")
    val constructorMirror = getCostructorMirror(cls)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams.map(_.size).sum)
    val params = T(Integer.valueOf(3),
      Integer.valueOf(32), Integer.valueOf(5), Integer.valueOf(5),
      Integer.valueOf(1), Integer.valueOf(1), Integer.valueOf(0),
      Integer.valueOf(0), Integer.valueOf(1), null.asInstanceOf[AnyRef],
      null.asInstanceOf[AnyRef], null.asInstanceOf[AnyRef], DataFormat("NCHW"))

//    ml.attribute.getDataType match {
//      case DataType.INT32 => Integer.valueOf(ml.attribute.getInt32Value)
//      case DataType.INT64 => Long.box(ml.attribute.getInt64Value)
//      case DataType.DOUBLE => Double.box(ml.attribute.getDoubleValue)
//      case DataType.FLOAT => Float.box(ml.attribute.getFloatValue)
//      case DataType.STRING => ml.attribute.getStringValue
//      case DataType.BOOL => Boolean.box(ml.attribute.getBoolValue)
//

    var i = 1
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (ptype <:< universe.typeOf[ClassTag[_]]||
          ptype.typeSymbol == universe.typeOf[ClassTag[_]].typeSymbol) {
          args(i) = ManifestFactory.Float
        } else if (ptype <:< universe.typeOf[TensorNumeric[_]]
          || ptype.typeSymbol == universe.typeOf[TensorNumeric[_]].typeSymbol) {
          args(i) = TensorNumeric.NumericFloat
        } else {
          println(s"***** ${i}")
          val value = params.get(i).get
          args(i) = value
          i += 1
        }
        val tmp = 0
      })
    })
    constructorMirror.apply(args : _*).
      asInstanceOf[AbstractModule[Activity, Activity, Float]]

    println("done")
  }
}
