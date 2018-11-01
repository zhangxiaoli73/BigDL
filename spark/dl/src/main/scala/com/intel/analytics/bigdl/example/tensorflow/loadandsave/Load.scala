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

package com.intel.analytics.bigdl.example.tensorflow.loadandsave

import java.nio.ByteOrder

import com.intel.analytics.bigdl.nn.{Graph, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.tf.TensorflowLoader

/**
 * This example show how to load a tensorflow model defined in slim and
 * use it to do prediction
 */
object Load {
  def main(args: Array[String]): Unit = {
    // require(args.length == 1, "Please input the model path as the first argument")

//    val p = "/home/zhangli/workspace/vgg/model.pb"
//    val input = Seq("input_node")
//    val output = Seq("vgg_16/fc8/squeezed")

    val p = "/home/zhangli/workspace/vgg/lenet.pb"
    val input = Seq("input_node")
    val output = Seq("LeNet/pool2/MaxPool")

//    val model = Module.loadTF(p, Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))

//    val model = Module.loadTF(args(0), Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))
//    val result = model.forward(Tensor(1, 1, 28, 28).rand())
//    println(result)

    val in = Tensor[Float](1, 32, 32, 3).rand()
    val model = TensorflowLoader.load(p, input, output, ByteOrder.LITTLE_ENDIAN)

    val m = model.asInstanceOf[Graph[Float]].modules

    val modelDef = TensorflowLoader.loadToIR(p, input, output)
    modelDef.build()

    val out1 = model.forward(in)
    val out2 = modelDef.forward(in)

//    val result = model.forward(Tensor(1, 1, 28, 28).rand())
//    println(result)

    val tmp = 0
  }
}
