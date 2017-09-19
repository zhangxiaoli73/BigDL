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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

@SerialVersionUID(- 8176191554025511686L)
class LSTMP[T : ClassTag] (
   val inputSize: Int,
   val hiddenSize: Int,
   val outputSize: Int,
   var wRegularizer: Regularizer[T] = null,
   var uRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val rate: Int = 0
 )(implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(hiddenSize, hiddenSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)
  ) {
  var gates: Sequential[T] = _
  var cellLayer: Sequential[T] = _
  override var cell: AbstractModule[Activity, Activity, T] = Sequential()
    .add(FlattenTable())
    .add(buildLSTM())
    .add(ConcatTable()
      .add(SelectTable(1))
      .add(NarrowTable(2, 2)))

  override def preTopology: AbstractModule[Activity, Activity, T] =
    TimeDistributed[T](Linear(inputSize, 4 * hiddenSize,
      wRegularizer = wRegularizer, bRegularizer = bRegularizer))

  override def hiddenSizeOfPreTopo: Int = 4 * hiddenSize

  def buildGates()(input1: ModuleNode[T], input2: ModuleNode[T])
  : (ModuleNode[T], ModuleNode[T], ModuleNode[T], ModuleNode[T]) = {

    var i2g: ModuleNode[T] = null
    var h2g: ModuleNode[T] = null

    i2g = input1
    h2g = Linear(hiddenSize, 4 * hiddenSize,
      withBias = false, wRegularizer = uRegularizer).inputs(input2)

    val caddTable = CAddTable(false).inputs(i2g, h2g)
    val reshape = Reshape(Array(4, hiddenSize)).inputs(caddTable)
    val split1 = Select(2, 1).inputs(reshape)
    val split2 = Select(2, 2).inputs(reshape)
    val split3 = Select(2, 3).inputs(reshape)
    val split4 = Select(2, 4).inputs(reshape)

    (Sigmoid().inputs(split1),
      Tanh().inputs(split2),
      Sigmoid().inputs(split3),
      Sigmoid().inputs(split4))
  }

  def buildLSTM(): Graph[T] = {
    val input1 = Input()
    val input2 = Input()
    val input3 = Input()
    val (in, hid, forg, out) = buildGates()(input1, input2)

    /**
      * g: Tanh
      * cMult1 = in * hid
      * cMult2 = forg * input3
      * cMult3 = out * g(cMult1 + cMult2)
      */
    val cMult1 = CMulTable().inputs(in, hid)
    val cMult2 = CMulTable().inputs(forg, input3)
    val cadd = CAddTable(true).inputs(cMult1, cMult2)
    val tanh = Tanh().inputs(cadd)
    val cMult3 = CMulTable().inputs(tanh, out)

    val out1 = Identity().inputs(cMult3)
    val out2 = Identity().inputs(cMult3)
    val out3 = cadd

    /**
      * out1 = cMult3
      * out2 = out1
      * out3 = cMult1 + cMult2
      */
    Graph(Array(input1, input2, input3), Array(out1, out2, out3))
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[LSTMP[T]]

  override def equals(other: Any): Boolean = other match {
    case that: LSTMP[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        inputSize == that.inputSize &&
        hiddenSize == that.hiddenSize &&
        outputSize == that.outputSize
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), inputSize, hiddenSize, outputSize)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def reset(): Unit = {
    super.reset()
    cell.reset()
  }


  override def toString: String = s"LSTM($inputSize, $hiddenSize, $outputSize)"
}

object LSTMP {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      inputSize: Int,
                                                      hiddenSize: Int,
                                                      outputSize: Int = 0,
                                                      wRegularizer: Regularizer[T] = null,
                                                      uRegularizer: Regularizer[T] = null,
                                                      bRegularizer: Regularizer[T] = null,
                                                      rate: Int = 0
                                                    )
                                                    (implicit ev: TensorNumeric[T]): LSTMP[T] = {
    new LSTMP[T](inputSize, hiddenSize, outputSize, wRegularizer, uRegularizer, bRegularizer)
  }
}
