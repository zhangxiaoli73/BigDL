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

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.{Graph, Reshape, StaticGraph}
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, Table}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor


class IRgraphSpec extends BigDLSpecHelper {
  def model() : Graph[Float] = {
    val conv1 = nn.SpatialConvolution(1, 20, 5, 5).inputs()
    val bn1 = nn.SpatialBatchNormalization(20).inputs(conv1)
    val pool1 = nn.SpatialMaxPooling(2, 2, 2, 2).setName("pool").inputs(bn1)
    val conv2 = nn.SpatialConvolution(20, 50, 5, 5).inputs(pool1)
    val pool2 = nn.SpatialMaxPooling(2, 2, 2, 2).inputs(conv2)
//    val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
//    val fc = nn.Linear(50 * 4 * 4, 500).inputs(reshape)
//    val relu = nn.ReLU().setName("relu1").inputs(fc)
//    val fc2 = nn.Linear(500, 10).setName("ip2").inputs(relu)
    val output = pool2
    Graph(conv1, output)
  }
 "static graph to ir graph & blas" should "be correct" in {
   val m = model().asInstanceOf[StaticGraph[Float]]

   val ir = m.toIRgraph()
   ir.build()

   val input = Tensor[Float](2, 1, 28, 28).rand()

   val p1 = ir.graph.getParametersTable()
   val p2 = m.getParametersTable()
   val keys = p1.keySet
   for (i <- keys) {
     val k = i.asInstanceOf[String]
     val t1 = p1[Table](k)
     val t2 = p2[Table](k)
     val w1 = t1[Tensor[Float]]("weight")
     val b1 = t1[Tensor[Float]]("bias")
     val gw1 = t1[Tensor[Float]]("gradWeight")
     val gb1 = t1[Tensor[Float]]("gradBias")

     val w2 = t2[Tensor[Float]]("weight")
     val b2 = t2[Tensor[Float]]("bias")
     val gw2 = t2[Tensor[Float]]("gradWeight")
     val gb2 = t2[Tensor[Float]]("gradBias")

     w1.copy(w2)
     b1.copy(b2)
     gw1.copy(gw2)
     gb1.copy(gb2)
    // t1 should be(t2)
   }

   val out1 = ir.forward(input)
   val out2 = m.forward(input)

   out1 should be(out2)
   println("done")
 }

  "static graph to ir graph & dnn" should "be correct" in {
    // System.setProperty("bigdl.engineType", "mkldnn")

    val m = model().asInstanceOf[StaticGraph[Float]]

    val ir = m.toIRgraph()
    ir.build()

    val input = Tensor[Float](2, 1, 28, 28).rand()

    val p1 = ir.graph.getParametersTable()
    val p2 = m.getParametersTable()
    val keys = p1.keySet
    for (i <- keys) {
      val k = i.asInstanceOf[String]
      val t1 = p1[Table](k)
      val t2 = p2[Table](k)
      val w1 = t1[Tensor[Float]]("weight")
      val b1 = t1[Tensor[Float]]("bias")
      val gw1 = t1[Tensor[Float]]("gradWeight")
      val gb1 = t1[Tensor[Float]]("gradBias")

      val w2 = t2[Tensor[Float]]("weight")
      val b2 = t2[Tensor[Float]]("bias")
      val gw2 = t2[Tensor[Float]]("gradWeight")
      val gb2 = t2[Tensor[Float]]("gradBias")

      w1.copy(w2)
      b1.copy(b2)
      gw1.copy(gw2)
      gb1.copy(gb2)
      // t1 should be(t2)
    }

    val out1 = ir.forward(input)
    val out2 = m.forward(input)

    out1 should be(out2)
    println("done")
  }
}
