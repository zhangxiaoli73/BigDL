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

import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec.P
import org.apache.spark.sql.catalyst.expressions.aggregate.First
import scopt.OptionParser

object MklPerf {

  val parser = new OptionParser[LocalPerfParam]("BigDL MKL Performance Test") {
    opt[Int]('m', "m")
      .text("m")
      .action((v, p) => p.copy(m = v))
    opt[Int]('n', "n")
      .text("n")
      .action((v, p) => p.copy(n = v))
    opt[Int]('k', "k")
      .text("k")
      .action((v, p) => p.copy(k = v))
    opt[Double]('a', "alpha")
      .text("alpha")
      .action((v, p) => p.copy(alpha = v.toFloat))
    opt[Double]('b', "beta")
      .text("beta")
      .action((v, p) => p.copy(beta = v.toFloat))
    opt[Boolean]('t', "trans")
      .text("trans")
      .action((v, p) => p.copy(trans = v))
    opt[Int]('t', "times")
      .text("times")
      .action((v, p) => p.copy(times = v))
    opt[String]('c', "computeType")
      .text("computeType")
      .action((v, p) => p.copy(computeType = v))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new LocalPerfParam()).foreach( param => {
      val m: Int = param.m
      val k: Int = param.k
      val n: Int = param.n

      val a: Array[Float] = new Array[Float](m * k)
      val b: Array[Float] = new Array[Float](k * n)
      val c: Array[Float] = new Array[Float](m * n)

      for (i <- 1 to m*k) {
        a(i-1) = RNG.uniform(0, 1).toFloat
      }
      for (i <- 1 to k*n) {
        b(i-1) = RNG.uniform(0, 1).toFloat
      }
      for (i <- 1 to m*n) {
        c(i-1) = RNG.uniform(0, 1).toFloat
      }

      if (param.computeType == "pack") {
        testPack(a, b, c, m, n, k, param)
      } else {
        testunPack(a, b, c, m, n, k, param)
      }
    })
  }

  def testPack(a: Array[Float], b: Array[Float], c: Array[Float], m: Int, n: Int, k: Int,
           param: LocalPerfParam): Unit = {
    val alpha: Float = param.alpha
    val beta: Float = param.beta
    val lda: Int = m
    val ldb: Int = k
    val ldc: Int = m

    val Second = if (param.trans) 'T' else 'N'

    val packMem: Long = MKL.sgemmAlloc('A', m, n, k)
    MKL.sgemmPack('A', Second, m, n, k, alpha, a, 0, lda, packMem)

    // warm up
    for (i <- 1 to 20) {
      MKL.sgemmCompute('P', 'N', m, n, k, a, 0, lda, b, 0, ldb, beta, c, 0, ldc, packMem)
    }

    println("pack start run")

    val t1 = System.nanoTime()
    for (i <- 1 to param.times) {
      MKL.sgemmCompute('P', 'N', m, n, k, a, 0, lda, b, 0, ldb, beta, c, 0, ldc, packMem)
    }
    val end1 = System.nanoTime() - t1

    MKL.sgemmFree(packMem)

    val number = System.getProperty("bigdl.mklNumThreads", "1")
    println(s"mklNumThreads $number and pack with sgemmCompute ${end1/1e9}, do tranpose ${param.trans}")
  }

  def testunPack(a: Array[Float], b: Array[Float], c: Array[Float], m: Int, n: Int, k: Int,
               param: LocalPerfParam): Unit = {
    val alpha: Float = 1f
    val beta: Float = 1f
    val lda: Int = m
    val ldb: Int = k
    val ldc: Int = m

    val First = if (param.trans) 'T' else 'N'

    // warm up
    for (i <- 1 to 20) {
      MKL.vsgemm(First, 'N', m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc)
    }

    println("unpack start run")
    val t1 = System.nanoTime()
    for (i <- 1 to param.times) {
      MKL.vsgemm(First, 'N', m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc)
    }
    val end1 = System.nanoTime() - t1

    val number = System.getProperty("bigdl.mklNumThreads", "1")
    println(s"mklNumThreads $number and unpack with vsgemm ${end1/1e9}, do tranpose ${param.trans}")
  }

}

case class LocalPerfParam(
    m : Int = 2600,
    k : Int = 650,
    n : Int = 1,
    times : Int = 30,
    beta : Float = 0.0f,
    alpha : Float = 1.0f,
    trans : Boolean = true,
    computeType : String = "pack" // another is "unpack"
  )