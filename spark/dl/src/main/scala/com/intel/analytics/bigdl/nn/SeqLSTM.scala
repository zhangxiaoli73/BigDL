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

import breeze.linalg.*
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.ContainerSerializable
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.sql.catalyst.expressions.If
import spire.syntax.module

import scala.reflect.ClassTag
import spire.macros.Auto.scala

import _root_.scala.collection.mutable.ArrayBuffer

class SeqLSTM[T: ClassTag] (inputSize: Int, hiddenSize: Int, var outputSize: Int = -1)
(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable{
  if (outputSize < 0) outputSize = hiddenSize

  val D = inputSize
  val H = hiddenSize
  val R = outputSize

  val weight = Tensor[T](D + R, 4 * H)
  val gradWeight = Tensor[T](D + R, 4 * H)
  val bias = Tensor[T](4 * H)
  val gradBias = Tensor[T](4 * H).zero()

  reset()

  val cell = Tensor[T]()  // this will be (T, N, H)
  var gates = Tensor[T]() // this will be (N, 4H)
  var gatesArr : Array[Tensor[T]] = null

  val buffer1 = Tensor[T]() // this will be (N, H)
  val buffer2 = Tensor[T]() // this will be (N, H)
  val buffer3 = Tensor[T]() // this will be (1, 4H)

  val grad_a_buffer = Tensor[T]() // this will be (N, 4H)
  val grad_buffer1 = Tensor[T]() // this will be (N, H)
  val grad_buffer2 = Tensor[T]()
  val grad_buffer3 = Tensor[T]()
  val grad_buffer4 = Tensor[T]()


  val h0 = Tensor[T]()
  val c0 = Tensor[T]()
  val _output = Tensor[T]()
  val _gradInput = Tensor[T]()

  val _remember = "neither"
  val batchfirst = true
  private var batchDim = 2
  private var timeDim = 1
  private val outputBuffer = Tensor[T]()
  private val gradBuffer = Tensor[T]()
  val preMM = Tensor[T]()

  if (batchfirst) {
    batchDim = 1
    timeDim = 2
  }
  override def reset(): Unit = {
    val std = 1.0 / math.sqrt(outputSize + inputSize)
    bias.zero()
    bias.narrow(1, outputSize + 1, outputSize).fill(ev.one)
    weight.randn(0, std)
  }

  def resetStates(): Unit = {
    h0.zero()
    c0.zero()
  }

  def checkDims(x: Tensor[T], dims: Array[Int]): Unit = {
    require(x.dim() == dims.length)
    var i = 0
    while (i < dims.length) {
      require(x.size(i + 1) == dims(i))
      i += 1
    }
  }

  def prepareSize(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 3)
    checkDims(input, Array(input.size(1), input.size(2), inputSize))
    checkDims(gradOutput, Array(input.size(1), input.size(2), outputSize))
  }

  def sigmoid(src: Tensor[T]): Tensor[T] = {
    val buffer = Tensor[T]().resizeAs(src).fill(ev.one)
    src.mul(ev.fromType(-1))
    src.exp().add(ev.one)
    buffer.cdiv(src)

    src.copy(buffer)
    src
  }

  def combine(src: Array[Int], target: Array[Int]): Unit = {
    require(src.length == target.length + 1,
      "TimeDistributed: combine method requires src.length == target.length + 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")

    target(0) = src(0) * src(1)
    var j = 1
    while (j < target.length) {
      target(j) = src(j + 1)
      j += 1
    }
  }

  def split(src: Array[Int], target: Array[Int], dim1: Int, dim2: Int): Unit = {
    require(src.length == target.length - 1,
      "TimeDistributed: split method requires src.length == target.length - 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")
    require(dim1 * dim2 == src(0),
      "TimeDistributed: split method requires dim1 * dim2 == src(0), " +
        s"Current dim1 = ${dim1}, dim2 = ${dim2}, src(0) = ${src(0)}")

    target(0) = dim1
    target(1) = dim2
    var j = 1
    while (j < src.length) {
      target(j + 1) = src(j)
      j += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val T = input.size(timeDim)
    val N = input.size(batchDim)

    val remember = false

    if (gatesArr == null) {
      gatesArr = new Array[Tensor[T]](T * 4)
      var i = 0
      while (i < gatesArr.length) {
        gatesArr(i) = Tensor[T](N, H)
        i += 1
      }
    }

    if (c0.nElement() == 0 || !remember) {
      c0.resize(N, H).zero()
    } else if (remember) {
      val prev_T = cell.size(1)
      val prev_N = cell.size(2)
      require(prev_N == N)
      c0.copy(cell.select(1, prev_T))
    }

    if (h0.nElement() == 0  || !remember) {
      h0.resize(N, R).zero()
    } else if (remember) {
      val prev_T = output.size(1)
      val prev_N = output.size(2)
      require(prev_N == N)
      c0.copy(output.select(1, prev_T))
    }

    // val bias_expand = bias.view(1, 4 * H).expand(Array(N, 4 * H))
    val bias_expand_test = bias.view(1, 4 * H).expand(Array(N * T, 4 * H))
    val Wx = weight.narrow(1, 1, D)
    val Wh = weight.narrow(1, D + 1, R)

    _output.resize(T, N, R).zero()
    cell.resize(T, N, R).zero()

    var prev_h : Tensor[T] = h0
    var prev_c : Tensor[T] = c0

   // TimeDistributed
    val t1 = System.nanoTime()
    val _inputSize = input.size
    val inputSize = new Array[Int](input.size.length - 1)
    val outputSize = new Array[Int](input.size.length)
    combine(_inputSize, inputSize)
    input.resize(inputSize)
    val tt1 = System.nanoTime()
    preMM.addmm(ev.one, bias_expand_test, ev.one, input, Wx)
    val tt2 = System.nanoTime()
    split(preMM.size, outputSize, _inputSize(0), _inputSize(1))
    preMM.resize(outputSize)
    input.resize(_inputSize)

    val t2 = System.nanoTime()
    gates.resize(N, 4 * H).zero()
    var t = 1
    while (t <= T) {
      val next_h = _output.select(1, t)
      val next_c = cell.select(1, t)

      Recurrent.selectCopy(preMM, t, gates)
      gates.addmm(prev_h, Wh)

      val i = gatesArr(t-1).copy(gates.narrow(2, 1, H)) // input gate
      val f = gatesArr(T + t - 1).copy(gates.narrow(2, H + 1, H)) // forget gate
      val o = gatesArr(2 * T + t - 1).copy(gates.narrow(2, 2 * H + 1, H)) // output gate
      val g = gatesArr(3 * T + t - 1).copy(gates.narrow(2, 3 * H + 1, H)) // input transform

      sigmoid(i)
      sigmoid(f)
      sigmoid(o)
      g.tanh()

      next_h.cmul(i, g)
      next_c.cmul(f, prev_c).add(next_h)
      next_h.tanh(next_c).cmul(o)

      // for LSTMP
      prev_h = next_h
      prev_c = next_c
      t += 1
    }
    val t3 = System.nanoTime()
    if (batchfirst) {
      Recurrent.transposeMemory(_output, output)
    }
    val t4 = System.nanoTime()
    // println(s"t1 ${(t2-t1)/1e9} t2 ${(t3 - t2)/1e9} t3 ${(t4-t3)/1e9} tt ${(tt2-tt1)/1e9}")
    output
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val st = System.nanoTime
    val scale = ev.fromType(1.0)

    val T = input.size(timeDim)
    val N = input.size(batchDim)

    val Wx = weight.narrow(1, 1, D)
    val Wh = weight.narrow(1, D + 1, R)
    val grad_Wx = gradWeight.narrow(1, 1, D)
    val grad_Wh = gradWeight.narrow(1, D + 1, R)

    _gradInput.resizeAs(input.transpose(1, 2)).zero()

    val grad_next_h = buffer1.resizeAs(h0).zero()
    val grad_next_c = buffer2.resizeAs(c0).zero()

    var t = T
    var prev_h : Tensor[T] = null
    var prev_c : Tensor[T] = null
//    while (t >= 1) {
//      val next_c = cell.select(1, t)
//      if (t == 1) {
//        prev_h = h0
//        prev_c = c0
//      } else {
//        prev_h = _output.select(1, t -1)
//        prev_c = cell.select(1, t - 1)
//      }
//
//      val gradTemp = if (batchfirst) {
//        Recurrent.selectCopy(gradOutput, t, gradBuffer)
//        gradBuffer
//      } else {
//        gradOutput.select(1, t)
//      }
//      grad_next_h.add(gradTemp)
//
//      val i = gatesArr(t-1)
//      val f = gatesArr(t-1 + T)
//      val o = gatesArr(t-1 + 2*T)
//      val g = gatesArr(t- 1 + 3*T)
//
//      val grad_a = grad_a_buffer.resize(N, 4 * H).zero()
//      val grad_ai = grad_buffer1.resize(N, H).zero()
//      val grad_af = grad_buffer2.resize(N, H).zero()
//      val grad_ao = grad_buffer3.resize(N, H).zero()
//      val grad_ag = grad_buffer4.resize(N, H).zero()
//
//      val tanh_next_c = grad_ai.tanh(next_c)
//      val tanh_next_c2 = grad_af.cmul(tanh_next_c, tanh_next_c)
//      val my_grad_next_c = grad_ao
//
//      my_grad_next_c.fill(ev.one)
//      my_grad_next_c.add(ev.fromType(-1), tanh_next_c2)
//      my_grad_next_c.cmul(o)
//      my_grad_next_c.cmul(grad_next_h)
//      grad_next_c.add(my_grad_next_c)
//
//      grad_ao.fill(ev.one).add(ev.fromType(-1), o).cmul(o).cmul(tanh_next_c).cmul(grad_next_h)
//      grad_a.narrow(2, 2 * H + 1, H).copy(grad_ao)
//
//      grad_ai.cmul(g, g)
//      grad_ag.fill(ev.one).add(ev.fromType(-1), grad_ai).cmul(i).cmul(grad_next_c)
//      grad_a.narrow(2, 3 * H + 1, H).copy(grad_ag)
//
//      grad_ai.fill(ev.one).add(ev.fromType(-1), i).cmul(i).cmul(g).cmul(grad_next_c)
//      grad_a.narrow(2, 1, H).copy(grad_ai)
//
//      grad_af.fill(ev.one).add(ev.fromType(-1), f).cmul(f).cmul(prev_c).cmul(grad_next_c)
//      grad_a.narrow(2, H + 1, H).copy(grad_af)
//
//      _gradInput.select(1, t).mm(grad_a, Wx.t())
//
//      val outTemp = if (batchfirst) {
//        Recurrent.selectCopy(input, t, outputBuffer)
//        outputBuffer
//      } else {
//        input.select(1, t)
//      }
//      grad_Wx.addmm(scale, outTemp.t(), grad_a)
//      grad_Wh.addmm(scale, prev_h.t(), grad_a)
//
//      grad_next_h.addmm(ev.zero, grad_next_h, ev.one, grad_a, Wh.t)
//      grad_next_c.cmul(f)
//
//      val grad_a_sum = buffer3.resize(1, 4 * H).sum(grad_a, 1)
//      gradBias.add(scale, grad_a_sum)
//
//      t = t -1
//    }

    grad_buffer1.resize(T, N, H).zero()
    grad_buffer1.tanh(cell)

    grad_buffer2.resize(T, N, H).zero()
    grad_buffer2.cmul(grad_buffer1, grad_buffer1)

    grad_buffer3.resize(T, N, H).fill(ev.one)
    grad_buffer3.add(ev.fromType(-1), grad_buffer2)

    while (t >= 1) {
      val next_c = cell.select(1, t)
      if (t == 1) {
        prev_h = h0
        prev_c = c0
      } else {
        prev_h = _output.select(1, t -1)
        prev_c = cell.select(1, t - 1)
      }

      val gradTemp = if (batchfirst) {
        Recurrent.selectCopy(gradOutput, t, gradBuffer)
        gradBuffer
      } else {
        gradOutput.select(1, t)
      }
      grad_next_h.add(gradTemp)

      val i = gatesArr(t-1)
      val f = gatesArr(t-1 + T)
      val o = gatesArr(t-1 + 2*T)
      val g = gatesArr(t- 1 + 3*T)

      val grad_a = grad_a_buffer.resize(N, 4 * H).zero()
      val grad_ai = grad_buffer1.select(1, t) // grad_buffer1.resize(N, H).zero()
      val grad_af = grad_buffer2.select(1, t) // grad_buffer2.resize(N, H).zero()
      val grad_ao = grad_buffer3.select(1, t) // grad_buffer3.resize(N, H).zero()
      val grad_ag = grad_buffer4.resize(N, H).zero()

//      val grad_ai = grad_buffer1.resize(N, H).zero()
//      val grad_af = grad_buffer2.resize(N, H).zero()
//      val grad_ao = grad_buffer3.resize(N, H).zero()
//      val grad_ag = grad_buffer4.resize(N, H).zero()
//
//       grad_ai.tanh(next_c)
//       grad_af.cmul(grad_ai, grad_ai)
//
//       grad_ao.fill(ev.one)
//       grad_ao.add(ev.fromType(-1), grad_af)

      grad_ao.cmul(grad_next_h)
      grad_ao.cmul(o)
      grad_next_c.add(grad_ao)

      grad_ao.fill(ev.one).add(ev.fromType(-1), o).cmul(o).cmul(grad_ai).cmul(grad_next_h)
      grad_a.narrow(2, 2 * H + 1, H).copy(grad_ao)

      grad_ai.cmul(g, g)
      grad_ag.fill(ev.one).add(ev.fromType(-1), grad_ai).cmul(i).cmul(grad_next_c)
      grad_a.narrow(2, 3 * H + 1, H).copy(grad_ag)

      grad_ai.fill(ev.one).add(ev.fromType(-1), i).cmul(i).cmul(g).cmul(grad_next_c)
      grad_a.narrow(2, 1, H).copy(grad_ai)

      grad_af.fill(ev.one).add(ev.fromType(-1), f).cmul(f).cmul(prev_c).cmul(grad_next_c)
      grad_a.narrow(2, H + 1, H).copy(grad_af)

      _gradInput.select(1, t).mm(grad_a, Wx.t())

      val outTemp = if (batchfirst) {
        Recurrent.selectCopy(input, t, outputBuffer)
        outputBuffer
      } else {
        input.select(1, t)
      }
      grad_Wx.addmm(scale, outTemp.t(), grad_a)
      grad_Wh.addmm(scale, prev_h.t(), grad_a)

      grad_next_h.addmm(ev.zero, grad_next_h, ev.one, grad_a, Wh.t)
      grad_next_c.cmul(f)

      val grad_a_sum = buffer3.resize(1, 4 * H).sum(grad_a, 1)
      gradBias.add(scale, grad_a_sum)

      t = t -1
    }
    if (batchfirst) {
      Recurrent.transposeMemory(_gradInput, gradInput)
    }
    this.backwardTime = System.nanoTime - st
    // println(s"backwardTime ${this.backwardTime/1e9}")
    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    backward(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {

  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }

  override def getParametersTable(): Table = {
    if (null == bias) {
      T(getName() -> T("weight" -> weight, "gradWeight" -> gradWeight))
    } else {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias))
    }
  }
}

object SeqLSTM extends ContainerSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    hiddenSize: Int,
    outputSize: Int = -1)
  (implicit ev: TensorNumeric[T]): SeqLSTM[T] = {
    new SeqLSTM[T](inputSize, hiddenSize, outputSize)
  }
}
