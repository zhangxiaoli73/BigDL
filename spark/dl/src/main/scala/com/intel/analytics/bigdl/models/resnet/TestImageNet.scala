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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.CropCenter
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, MTImageFeatureToBatch, MatToTensor, PixelBytesToMat}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelScaledNormalizer, RandomCropper, RandomResize}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.intermediate.{ConversionUtils, IRGraph}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
 * This example is to evaluate trained resnet50 with imagenet data and get top1 and top5 accuracy
 */
object TestImageNet {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils._

  def toNCHW(src: Tensor[Float], inputFormat: MemoryData): Tensor[Float] = {
    val outputFormat = HeapData(inputFormat.shape,
      if (src.size().length == 2) { Memory.Format.nc } else { Memory.Format.nchw })
    val reorder = ReorderMemory(inputFormat, outputFormat, null, null)

    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat))
    reorder.forward(src).toTensor
  }

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Test model on ImageNet2012")
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      import com.intel.analytics.bigdl.models.resnet
      val model = ResNet(classNum = 1000, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> false, "dataSet" -> DatasetType.ImageNet)).toGraph()
      // Module.loadModule[Float](param.model)
      val modelDnn = ConversionUtils.convert(model.cloneModule().asInstanceOf[StaticGraph[Float]]
        .setOutputFormats(Seq(Memory.Format.nchw)))

      val pool = SpatialMaxPooling[Float](3, 3, 2, 2, 1, 1)
//      val input = Tensor[Float](2, 3, 224, 224).rand()
//      val gradOutput = Tensor[Float](2, 1000).rand()
//      val out1 = model.forward(input).toTensor[Float]
//      val out2 = modelDnn.forward(input)
//      val grad1 = model.backward(input, gradOutput).toTensor[Float]
//      val grad2 = modelDnn.backward(input, gradOutput).toTensor[Float]
//
//      val bk = model.asInstanceOf[StaticGraph[Float]].getExecutions()
//      val bkDnn = modelDnn.asInstanceOf[IRGraph[Float]].graph.asInstanceOf[DnnGraph].getExecutions()
//
//
//      var j = 2
//      while (j < bk.length) {
//        if (bk(j).element.isInstanceOf[SpatialBatchNormalization[Float]]) {
//          val blas = bk(j).element
//          val dnn = bkDnn(j + 1).element
//          println("done")
//        }
//        j += 1
//      }
      val evaluationSet = ImageNetDataSet.valDataSet(param.folder,
        sc, 224, param.batchSize).toDistributed().data(train = false)

      val criterion = new CrossEntropyCriterion[Float]()
      val criterionDnn = new CrossEntropyCriterion[Float]()

      val data = evaluationSet.mapPartitions(batch => {
        batch.map(dd => {
          (dd.getInput().toTensor[Float].clone(), dd.getTarget().toTensor[Float].clone())
        })
      }).collect()

      val length = data.length
      var i = 0
      while (i < length) {
        RandomGenerator.RNG.setSeed(100)
        val input = Tensor[Float](8, 3, 224, 224).rand(-1, 1) // data(i)._1
        val target = data(i)._2

        val out1 = model.forward(input).toTensor[Float]
        val out2 = modelDnn.forward(input).toTensor[Float]

        Equivalent.getunequals(out1, out2, 1e-5)

//        criterion.forward(out1, target)
//        val cri = criterion.backward(out1, target)

//        RandomGenerator.RNG.setSeed(100)
        val cri = out1 // Tensor[Float]().resizeAs(out1).rand()
        val grad1 = model.backward(input, cri).toTensor[Float]
        val grad2 = modelDnn.backward(input, cri).toTensor[Float]

        Equivalent.getunequals(grad1, grad2, 1e-5)

        model.zeroGradParameters()
        modelDnn.zeroGradParameters()

        val bk = model.asInstanceOf[StaticGraph[Float]].getExecutions()
        val bkDnn = modelDnn.asInstanceOf[IRGraph[Float]].graph.asInstanceOf[DnnGraph].getExecutions()

        val diff = if (bkDnn.length > bk.length + 1) 1 else 0
        var j = 0
        while (j < bk.length) {
          val blas = bk(j).element
          val dnn = bkDnn(j + diff).element
          println(blas + "--------------" + dnn)

          val outBlas = blas.output.toTensor[Float]
          val outDnn = if (dnn.output.asInstanceOf[Tensor[Float]].isInstanceOf[DnnTensor[Float]]) {
            toNCHW(dnn.output.asInstanceOf[Tensor[Float]],
              dnn.asInstanceOf[MklDnnLayer].outputFormats()(0))
          } else {
            dnn.output.toTensor[Float]
          }

          val gradBlas = blas.gradInput.toTensor[Float]
          val gradDnn =
            if (dnn.gradInput.asInstanceOf[Tensor[Float]].isInstanceOf[DnnTensor[Float]]) {
            toNCHW(dnn.gradInput.asInstanceOf[Tensor[Float]],
              dnn.asInstanceOf[MklDnnLayer].gradInputFormats()(0))
          } else {
              dnn.gradInput.toTensor[Float]
          }
          Equivalent.getunequals(outBlas, outDnn, 1e-5)
          Equivalent.getunequals(gradBlas, gradDnn, 1e-5)
          println("done")
//
//          if (bk(j).element.isInstanceOf[SpatialBatchNormalization[Float]]) {
//            val blas = bk(j).element.asInstanceOf[SpatialBatchNormalization[Float]]
//            val dnn = bkDnn(j + 1).element.asInstanceOf[BlasWrapper].module.
//              asInstanceOf[SpatialBatchNormalization[Float]]
//
//            val inBlas = blas.inputBuffer
//            val inDnn = dnn.inputBuffer
//            Equivalent.nearequals(inBlas, inDnn)
//
//            val gradOutBlas = blas.gradOutputBuffer
//            val gradOutDnn = dnn.gradOutputBuffer
//            Equivalent.nearequals(gradOutBlas, gradOutDnn)
//
//            val outBlas = blas.output
//            val outDnn = dnn.output
//
//            val gradBlas = blas.gradInput
//            val gradDnn = dnn.gradInput
//
//            println("done")
//          }
          j += 1
        }

        if (i == 10) {
          val tmp = 0
        }
        i += 1
        println(s"******* ${i}")
      }
//      val result = model.evaluate(evaluationSet,
//        Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
//      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    })
  }
}
