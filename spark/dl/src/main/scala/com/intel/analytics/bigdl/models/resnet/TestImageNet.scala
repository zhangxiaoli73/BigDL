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

      RandomGenerator.RNG.setSeed(1)
      import com.intel.analytics.bigdl.models.resnet
      val seqModel = ResNet(classNum = 1000, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> false, "dataSet" -> DatasetType.ImageNet))
      ResNet.modelInit(seqModel)
      val model = seqModel.toGraph()
      // println(seqModel)
      val modelDnn = ConversionUtils.convert(model.cloneModule().asInstanceOf[StaticGraph[Float]]
        .setOutputFormats(Seq(Memory.Format.nc)))

      val evaluationSet = ImageNetDataSet.valDataSet(param.folder,
        sc, 224, param.batchSize).toDistributed().data(train = false)

      val criterion = new CrossEntropyCriterion[Float]()
      val criterionDnn = new CrossEntropyCriterion[Float]()

//      val data = evaluationSet.mapPartitions(batch => {
//        batch.map(dd => {
//          (dd.getInput().toTensor[Float].clone(), dd.getTarget().toTensor[Float].clone())
//        })
//      }).collect()

      var i = 0
      val length = 1 // data.length
      while (i < length) {
        RandomGenerator.RNG.setSeed(100)
        val input = Tensor[Float](8, 3, 224, 224).rand(-1, 1)
        // val target = data(i)._2

        val out1 = model.forward(input).toTensor[Float]
        val out2 = modelDnn.forward(input).toTensor[Float]

        Equivalent.getunequals(out1, out2, 1e-5)
//        criterion.forward(out1, target)
//        val cri = criterion.backward(out1, target)
         val cri = out1

        val grad1 = model.backward(input, cri).toTensor[Float]
        val grad2 = modelDnn.backward(input, cri).toTensor[Float]

        Equivalent.getunequals(grad1, grad2, 1e-5)

        model.zeroGradParameters()
        modelDnn.zeroGradParameters()

        val bk = model.asInstanceOf[StaticGraph[Float]].getExecutions()
        val bkDnn = modelDnn.asInstanceOf[IRGraph[Float]].graph.asInstanceOf[DnnGraph].getExecutions()

        val diff = if (bkDnn.length > bk.length + 1) 1 else 0
        var j = 0
        while (j < bk.length - 1) {
          val blas = bk(j).element
          val dnn = bkDnn(j + diff).element
          if (!blas.asInstanceOf[Module[Float]].isInstanceOf[nn.CAddTable[Float, _]]) {
            println(blas + "--------------" + dnn)
            println(dnn.asInstanceOf[MklDnnLayer].outputFormats()(0) + " "
              + dnn.asInstanceOf[MklDnnLayer].gradInputFormats()(0))

            val outBlas = blas.output.toTensor[Float]
            val outDnn = if (dnn.output.asInstanceOf[Tensor[Float]]
              .isInstanceOf[DnnTensor[Float]]) {
              toNCHW(dnn.output.asInstanceOf[Tensor[Float]],
                dnn.asInstanceOf[MklDnnLayer].outputFormats()(0))
            } else {
              dnn.output.toTensor[Float]
            }

            val gradBlas = blas.gradInput.toTensor[Float]
            val gradDnn = if (dnn.gradInput.asInstanceOf[Tensor[Float]]
              .isInstanceOf[DnnTensor[Float]]) {
              toNCHW(dnn.gradInput.asInstanceOf[Tensor[Float]],
                dnn.asInstanceOf[MklDnnLayer].gradInputFormats()(0))
            } else {
              dnn.gradInput.toTensor[Float]
            }

            println("output difference")
            val num = Equivalent.getunequals(outBlas, outDnn, 1e-4)
            val outputElment = num.toFloat / outBlas.nElement()
            if (outputElment > 0.05) {
              val tmp = 0
            }
            println("gradInput difference")
            val numGrad = Equivalent.getunequals(gradBlas, gradDnn, 1e-4)
            val gradElment = numGrad.toFloat / gradBlas.nElement()
            if (gradElment > 0.05) {
              Equivalent.getunequals(gradBlas, gradDnn, 1e-3, debug = false)
              val tmp = 0
            }
            //            if (dnn.getName() == "res2b_branch2a") {
            //              val gradDnn =
            //                if (dnn.gradInput.asInstanceOf[Tensor[Float]].isInstanceOf[DnnTensor[Float]]) {
            //                  val tmp = dnn.asInstanceOf[MklDnnLayer].gradInputFormats()(0)
            //                toNCHW(dnn.gradInput.asInstanceOf[Tensor[Float]], NativeData(tmp.shape, Memory.Format.nchw))
            //              } else {
            //                dnn.gradInput.toTensor[Float]
            //              }
            //              Equivalent.getunequals(gradBlas, gradDnn, 1e-4)
            //            } else {
            //              Equivalent.getunequals(gradBlas, gradDnn, 1e-4)
            //            }
            //            if (dnn.getName() == "res2b_branch2a") {
            //              val in1Buffer = blas.asInstanceOf[nn.SpatialConvolution[Float]].inputBuffer
            //              val grad1Buffer = blas.asInstanceOf[nn.SpatialConvolution[Float]].gradOutputBuffer
            //
            //              val in2Buffer = toNCHW(dnn.asInstanceOf[nn.mkldnn.SpatialConvolution].inputBuffer,
            //                  dnn.asInstanceOf[MklDnnLayer].inputFormats()(0)).clone()
            //              val grad2Buffer = toNCHW(dnn.asInstanceOf[nn.mkldnn.SpatialConvolution].gradOutputBuffer,
            //                  dnn.asInstanceOf[MklDnnLayer].gradOutputFormats()(0)).clone()
            //
            //              Equivalent.getunequals(in1Buffer, in2Buffer, 1e-5)
            //              Equivalent.getunequals(grad1Buffer, grad2Buffer, 1e-4)
            //
            //              val conv = resnet.Convolution[Float](256, 64, 1, 1, 1, 1, 0, 0, optnet = false)
            //              // resnet.Convolution[Float](3, 64, 7, 7, 2, 2, 3, 3, optnet = false)
            //              val p1 = blas.getParameters()
            //              val p2 = conv.getParameters()
            //
            //              p2._1.copy(p1._1)
            //              p2._2.copy(p1._2)
            //
            //              val outTmp = conv.forward(in2Buffer).clone()
            //              val gradTmp = conv.updateGradInput(in2Buffer, grad2Buffer).clone()
            //              Equivalent.getunequals(gradTmp, gradDnn, 1e-4)
            //
            //              val outTmp2 = conv.forward(in1Buffer).clone()
            //              val gradTmp2 = conv.updateGradInput(in1Buffer, grad1Buffer).clone()
            //              Equivalent.getunequals(gradTmp2, gradBlas, 1e-4)
            //
            //              Equivalent.getunequals(gradTmp2, gradTmp, 1e-4)
            //
            //              println("done")
            //            }
            println(s"${dnn} done")
          }
          j += 1
        }

        println(s"********** ${i}")
        i += 1
      }
      sc.stop()
    })
  }
}
