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

package com.intel.analytics.bigdl.nn.mkldnn.models

import java.io.{File, PrintWriter}
import java.nio.file.{Files, Paths}

import breeze.numerics.abs
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.mkldnn.{HeapData, MemoryData, MklDnnRuntime, ReorderMemory}
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Tensor}

import scala.sys.process._

object Tools {
  def compare2Tensors(src: Tensor[Float], dst: Tensor[Float]): Boolean = {
    Equivalent.nearequals(dense(src).toTensor, dense(dst).toTensor)
  }

  def dense(t: Activity): Activity = {
    val ret = if (t.isTensor) {
      val tt = t.asInstanceOf[Tensor[Float]]
      Tensor[Float]().resize(tt.size()).copy(tt)
    } else {
      throw new UnsupportedOperationException
    }

    ret
  }

  def toNCHW(src: Tensor[Float], inputFormat: MemoryData): Tensor[Float] = {
    val outputFormat = HeapData(inputFormat.shape,
      if (src.size().length == 2) { Memory.Format.nc } else { Memory.Format.nchw })
    val reorder = ReorderMemory(inputFormat, outputFormat, null, null)

    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.forward(src).toTensor
  }

  def toNCHWShape(src: Tensor[Float], inputFormat: MemoryData,
                  outputShape: Array[Int]): Tensor[Float] = {
    val outputFormat = HeapData(outputShape,
      if (src.size().length == 2) { Memory.Format.nc } else { Memory.Format.nchw })
    val reorder = ReorderMemory(inputFormat, outputFormat, null, null)

    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.forward(src).toTensor
  }

  def fromNCHW(src: Tensor[Float], outputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = src.size().length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.nc
      case 4 => Memory.Format.nchw
    }

    val inputFormat = HeapData(src.size(), defaultFormat)
    val reorder = ReorderMemory(inputFormat, outputFormat, null, null)
    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.forward(src).toTensor
  }

  def fromOIHW(src: Tensor[Float], outputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = outputFormat.shape.length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.oi
      case 4 => Memory.Format.oihw
    }

    val inputFormat = HeapData(outputFormat.shape, defaultFormat)
    val reorder = ReorderMemory(inputFormat, outputFormat, null, null)
    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.updateOutput(src).toTensor
  }

  def toOIHW(src: Tensor[Float], inputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = inputFormat.shape.length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.oi
      case 4 => Memory.Format.oihw
      case 5 => Memory.Format.goihw
    }

    val outputFormat = HeapData(inputFormat.shape, defaultFormat)
    val reorder = ReorderMemory(inputFormat, outputFormat, null, null)
    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.updateOutput(src).toTensor
  }
}

/**
 * Call "collect" command, which is a method to collect output binary files.
 * It's similar to "caffe collect", the difference is that it supports collect
 * single layers output and gradient through make a fake gradOutput/top_diff.
 */
object Collect {
  val tmpdir: String = System.getProperty("java.io.tmpdir")
  val collectPath: String = System.getProperty("collect.location")

  def hasCollect: Boolean = {
    val exitValue = if (collectPath != null) s"ls $collectPath".! else "which collect".!
    exitValue == 0
  }

  /**
   * save the prototxt to a temporary file and call collect
   * @param prototxt prototxt with string
   * @return the middle random number in temporary file, which is an identity for getTensor.
   */
  def run(prototxt: String, singleLayer: Boolean = true): String = {
    def saveToFile(prototxt: String, name: String): String = {
      val tmpFile = java.io.File.createTempFile(name, ".prototxt")
      val absolutePath = tmpFile.getAbsolutePath

      println(s"prototxt is saved to $absolutePath")

      val writer = new PrintWriter(tmpFile)
      writer.println(prototxt)
      writer.close()

      absolutePath
    }

    if (! hasCollect) {
      throw new RuntimeException(s"Can't find collect command. Have you copy to the PATH?")
    }

    val file = saveToFile(prototxt, "UnitTest.") // UnitTest ends with dot for getting random number
    val identity = file.split("""\.""").reverse(1) // get the random number

    val cmd = Seq(s"$collectPath", "--model", file, "--type", "float", "--identity", identity)
    val exitValue = if (singleLayer) {
      Process(cmd :+ "--single", new File(tmpdir)).!
    } else {
      Process(cmd, new File(tmpdir)).!
    }

    Files.deleteIfExists(Paths.get(file))
    require(exitValue == 0, s"Something wrong with collect command. Please check it.")

    identity
  }
}

object Utils {
  def time[R](block: => R): (Double, R) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val takes = (t1 - t0) / 1e9
    (takes, result)
  }

  def manyTimes[R](block: => R)(iters: Int): (Double, R) = {
    time[R] {
      var i = 0
      while (i < iters - 1) {
        block
        i += 1
      }
      block
    }
  }

  def speedup(base: Double, after: Double): String = {
    val result = (base - after) / base
    ((result * 1000).toInt / 10.0).toString + "%"
  }
}

object Equivalent {

  def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
    val absA = math.abs(a)
    val absB = math.abs(b)
    val diff = math.abs(a - b)

    val result = if (a == b) {
      true
    } else {
      math.min(diff / (absA + absB), diff) < epsilon
    }

    result
  }

  def nearequals(t1: Tensor[Float], t2: Tensor[Float],
    epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    result
  }

  def getunequals(t1: Tensor[Float], t2: Tensor[Float],
    epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    var num = 0
    t1.map(t2, (a, b) => {
      if (true) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          num += 1
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    println("diff num " + num)
    return true
  }

  def isEquals(t1: Tensor[Float], t2: Tensor[Float]): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = if (a == b) true else false
        if (!result) {
          val diff = Math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    return result
  }
}
