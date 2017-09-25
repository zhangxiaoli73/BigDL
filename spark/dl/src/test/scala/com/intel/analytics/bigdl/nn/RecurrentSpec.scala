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

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.torch.TH
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class RecurrentSpec extends FlatSpec with Matchers {

  "A Cell class " should "call addTimes() correctly" in {
    val hiddenSize = 5
    val inputSize = 5
    val outputSize = 5
    val batchSize = 5
    val time = 4
    val seed = 100
    RNG.setSeed(seed)
    val rnnCell1 = RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())
    val rnnCell2 = RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())
    val rnnCell3 = RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())
    val rnnCell4 = RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())

    val input = Tensor[Double](batchSize, inputSize).randn
    val hidden = Tensor[Double](batchSize, hiddenSize).randn
    val gradOutput = Tensor[Double](batchSize, outputSize).randn
    val gradHidden = Tensor[Double](batchSize, outputSize).randn

    rnnCell1.forward(T(input, hidden))
    rnnCell1.backward(T(input, hidden), T(gradOutput, gradHidden))
    rnnCell2.forward(T(input, hidden))
    rnnCell2.backward(T(input, hidden), T(gradOutput, gradHidden))
    rnnCell3.forward(T(input, hidden))
    rnnCell3.backward(T(input, hidden), T(gradOutput, gradHidden))
    rnnCell4.forward(T(input, hidden))
    rnnCell4.backward(T(input, hidden), T(gradOutput, gradHidden))

    val forwardSum = new Array[Long](6)
    val backwardSum = new Array[Long](6)

    for (i <- 0 until 6) {
      forwardSum(i) += rnnCell1.getTimes()(i)._2
      backwardSum(i) += rnnCell1.getTimes()(i)._3
    }
    for (i <- 0 until 6) {
      forwardSum(i) += rnnCell2.getTimes()(i)._2
      backwardSum(i) += rnnCell2.getTimes()(i)._3
    }
    for (i <- 0 until 6) {
      forwardSum(i) += rnnCell3.getTimes()(i)._2
      backwardSum(i) += rnnCell3.getTimes()(i)._3
    }
    for (i <- 0 until 6) {
      forwardSum(i) += rnnCell4.getTimes()(i)._2
      backwardSum(i) += rnnCell4.getTimes()(i)._3
    }

    rnnCell1.addTimes(rnnCell2)
    rnnCell1.addTimes(rnnCell3)
    rnnCell1.addTimes(rnnCell4)

    for (i <- 0 until 6) {
      forwardSum(i) should be (rnnCell1.getTimes()(i)._2)
      backwardSum(i) should be (rnnCell1.getTimes()(i)._3)
    }
  }

  "A Recurrent" should " call getTimes correctly" in {
    val hiddenSize = 128
    val inputSize = 1280
    val outputSize = 128
    val time = 30
    val batchSize1 = 100
    val batchSize2 = 8
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()

        .add(LSTM[Double](inputSize, hiddenSize)))
      .add(Select(2, 1))
    //      .add(Linear[Double](hiddenSize, outputSize))

    val input = Tensor[Double](Array(batchSize1, time, inputSize)).rand
    val gradOutput = Tensor[Double](batchSize1, outputSize).rand

    model.clearState()

    model.resetTimes
    model.getTimes

    for (i <- 1 to 10) {
      model.resetTimes
      model.forward(input)
      model.backward(input, gradOutput)
      model.getTimes()
    }
    model.resetTimes()

    var st = System.nanoTime()
    model.forward(input)
    val etaForward = System.nanoTime() - st
    println(s"forward eta = ${etaForward}")
    st = System.nanoTime()
    model.backward(input, gradOutput)
    val etaBackward = System.nanoTime() - st
    println(s"backward eta = ${etaBackward}")
    println()
    var forwardSum = 0L
    var backwardSum = 0L

    model.getTimes.foreach(x => {
      println(x._1 + ", " + x._2 + ", " + x._3)
      forwardSum += x._2
      backwardSum += x._3
    })
    println()
    println(s"forwardSum = ${forwardSum}")
    println(s"backwardSum = ${backwardSum}")

    assert(abs((etaForward - forwardSum) / etaForward) < 0.1)
    assert(abs((etaBackward - backwardSum) / etaBackward) < 0.1)
  }

  "A Recurrent with LSTMPeephole cell " should " add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val batchSize = 4
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(LSTMPeephole[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    println(model)

    val input = Tensor[Double](batchSize, time, inputSize)
    val gradOutput = Tensor[Double](batchSize, time, outputSize)

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    println("add normalization")
  }

  "A Recurrent with GRU cell " should " add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val batchSize = 4
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(GRU[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    println(model)

    val input = Tensor[Double](batchSize, time, inputSize)
    val gradOutput = Tensor[Double](batchSize, time, outputSize)

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    println("add normalization")
  }

  "A Recurrent with LSTM cell " should " add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val batchSize = 4
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(LSTM[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    println(model)

    val input = Tensor[Double](batchSize, time, inputSize)
    val gradOutput = Tensor[Double](batchSize, time, outputSize)

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    println("add normalization")
  }

  "A Recurrent with SimpleRNN cell " should " add batchNormalization correctly" in {
    val hiddenSize = 4
    val inputSize = 5
    val batchSize = 2
    val time = 2
    val seed = 100
    RNG.setSeed(seed)

    val cell = RnnCell[Double](inputSize, hiddenSize, ReLU[Double]())
    val model = Sequential[Double]()
      .add(Recurrent[Double](BatchNormParams())
        .add(cell))

    val (weightsArray, gradWeightsArray) = model.parameters()
    weightsArray(0).set(Tensor[Double](Array(0.038822557355026155,
      0.15308625574211315, -0.1982324504512677, -0.07866809418407278,
      -0.06751351799422134, 0.023597193777786962, 0.3083771498964048,
      -0.31429738377130323, -0.4429929170091549, -0.30704694098520874,
      -0.33847886911170505, -0.2804322767460886, 0.15272262323432112,
      -0.2592875227066882, 0.2914515643266326, -0.0422707164265147,
      -0.32493950675524846, 0.3310656372548169, 0.06716552027607742, -0.39025554201755425),
      Array(4, 5)))
    weightsArray(1).set(Tensor[Double](Array(0.3500089930447004,
      0.11118793394460541,
      -0.2600975267200473,
      0.020882861472978465), Array(4)))

    weightsArray(2).set(Tensor[Double](Array(0.18532821908593178,
    0.5622962701600045,
    0.10837689251638949,
    0.005817196564748883),
      Array(4)))

    weightsArray(4).set(Tensor[Double](Array(-0.28030250454321504,
      -0.19257679535076022, 0.4786237839143723, 0.45018431078642607,
    0.31168314907699823, -0.37334575527347624, -0.3280589876230806, -0.4210121303331107,
    0.31622475129552186, -0.18864686344750226, -0.22592625673860312, 0.13238358590751886,
    -0.06829581526108086, 0.1993589240591973, 0.44002981553785503, 0.14196494384668767),
      Array(4, 4)))
    weightsArray(5).set(Tensor[Double](Array(0.3176493758801371,
    0.4200237800832838,
    -0.16388805187307298,
    -0.20112364063970745), Array(4)))

    val input = Tensor[Double](Array(0.1754104527644813, 0.5687455364968628,
      0.3728320465888828, 0.17862433078698814, 0.005688507109880447,
    0.5325737004168332, 0.2524263544473797, 0.6466914659831673, 0.7956625143997371,
      0.14206538046710193, 0.015254967380315065, 0.5813889650162309, 0.5988433782476932,
      0.4791899386327714, 0.6038045417517424, 0.3864191132597625, 0.1051476860884577,
      0.44046495063230395, 0.3819434456527233, 0.40475733182393014),
      Array(batchSize, time, inputSize))
    val gradOutput = Tensor[Double](Array(0.015209059891239357, 0.08723655440707856,
      1.2730716350771312, 0.17783007683002253, 0.9809208554215729,
      0.7760053128004074, 0.05994199030101299, 0.550958373118192,
    1.1344734990039382, -0.1642483852831349, 0.585060822398516, 0.6124844773937481,
    0.7424796849954873, 0.95687689865008, 0.6301839421503246, 0.17582130827941),
      Array(batchSize, time, hiddenSize))

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    output should be (Tensor[Double](Array(0.6299258968399799,
      1.3642297404555106, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0361454784986459,
    0.25696697120146866, 0.19273657953649884, 0.0, 0.0,
    0.12799813658263415, 0.24882574216093045, 0.0, 0.0),
      Array(batchSize, time, hiddenSize)))

    gradInput should be (Tensor[Double](Array(-0.027275798942804907,
      -0.1383686829541865, 0.16717516624801407, 0.11372249422239256,
      0.08955797331273728, -0.08873463347124873, -0.38487333246986216,
      0.48437215964262187, 0.2417856458208813, 0.19281229062491595,
    0.03225607513585589, -0.36656214055421815, 0.2795038794253451,
      0.8385161794048844, 0.549019363159085, 0.08375435727819777,
      0.8898041559782669, -0.9310512053159811, -1.1940243194481583, -0.8313896270967381),
      Array(batchSize, time, inputSize)))
  }

  "A Recurrent" should " converge when batchSize changes" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val time = 4
    val batchSize1 = 5
    val batchSize2 = 8
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())))
      .add(Select(2, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    val input1 = Tensor[Double](Array(batchSize1, time, inputSize)).rand
    val input2 = Tensor[Double](batchSize2, time, inputSize).rand

    val gradOutput1 = Tensor[Double](batchSize1, outputSize).rand
    val gradOutput2 = Tensor[Double](batchSize2, outputSize).rand

    model.clearState()

    model.forward(input1)
    model.backward(input1, gradOutput1)
    val gradInput1 =
      Tensor[Double](batchSize1, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output1 = Tensor[Double](batchSize1, outputSize).copy(model.output.toTensor[Double])

    model.clearState()

    model.forward(input2)
    model.backward(input2, gradOutput2)
    val gradInput2 =
      Tensor[Double](batchSize2, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output2 = Tensor[Double](batchSize2, outputSize).copy(model.output.toTensor[Double])

    model.forward(input1)
    model.backward(input1, gradOutput1)
    val gradInput1compare =
      Tensor[Double](batchSize1, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output1compare = Tensor[Double](batchSize1, outputSize)
      .copy(model.output.toTensor[Double])

    model.forward(input2)
    model.backward(input2, gradOutput2)
    val gradInput2compare =
      Tensor[Double](batchSize2, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output2compare = Tensor[Double](batchSize2, outputSize)
      .copy(model.output.toTensor[Double])

    model.hashCode()

    output1 should be (output1compare)
    output2 should be (output2compare)

    gradInput1 should be (gradInput1compare)
    gradInput2 should be (gradInput2compare)
  }

  "A Recurrent Language Model Module " should "converge" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0)*inputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0)*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    for (i <- 1 to 50) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A Recurrent Module " should "converge in batch mode" in {

    val batchSize = 10
    val nWords = 5
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh())))
      .add(Select(2, nWords))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()

    val input = Tensor[Double](Array(batchSize, nWords, inputSize))
    val labels = Tensor[Double](batchSize)
    for (b <- 1 to batchSize) {
      for (i <- 1 to nWords) {
        val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
      }
      val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0) * outputSize).toInt
      labels.setValue(b, rdmLabel)
    }

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    for (i <- 1 to 50) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A Recurrent Module " should "perform correct gradient check" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh())))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(Math.random()*inputSize).toInt
      val rdmInput = Math.ceil(Math.random()*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println("gradient check for input")
    val gradCheckerInput = new GradientChecker(1e-2, 1)
    val checkFlagInput = gradCheckerInput.checkLayer[Double](model, input)
    println("gradient check for weights")
    val gradCheck = new GradientCheckerRNN(1e-2, 1)
    val checkFlag = gradCheck.checkLayer(model, input, labels)
  }

  "Recurrent dropout" should "work correclty" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 1

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(1, seqLength, inputSize))
    for (i <- 1 to seqLength) {
      val rdmInput = 3
      input.setValue(1, i, rdmInput, 1.0)
    }

    println(input)
    val gru = GRU[Double](inputSize, hiddenSize, 0.2)
    val model = Recurrent[Double]().add(gru)

    val field = model.getClass.getDeclaredField("cells")
    field.setAccessible(true)
    val cells = field.get(model).asInstanceOf[ArrayBuffer[Cell[Double]]]

    val dropoutsRecurrent = model.asInstanceOf[Container[_, _, Double]].findModules("Dropout")
    val dropoutsCell = gru.cell.asInstanceOf[Container[_, _, Double]].findModules("Dropout")
    val dropouts = dropoutsRecurrent ++ dropoutsCell
    dropouts.size should be (6)

    val output = model.forward(input)
    val noises1 = dropouts.map(d => d.asInstanceOf[Dropout[Double]].noise.clone())
    noises1(0) should not be noises1(1)

    val noises = dropoutsCell.map(d => d.asInstanceOf[Dropout[Double]].noise.clone())
    for (i <- dropoutsCell.indices) {
      cells.foreach(c => {
        val noise = c.cell.asInstanceOf[Container[_, _, Double]]
          .findModules("Dropout")(i)
          .asInstanceOf[Dropout[Double]]
          .noise
        noise should be(noises(i))
      })
    }


    model.forward(input)

    var flag = true
    for (i <- dropoutsCell.indices) {
      cells.foreach(c => {
        val newNoises = c.cell.asInstanceOf[Container[_, _, Double]]
          .findModules("Dropout")
        val noise = newNoises(i).asInstanceOf[Dropout[Double]].noise
        flag = flag && (noise == noises(i))
      })
    }

    flag should be (false)
  }

  "A Recurrent Module " should "work with get/set state " in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    val batchSize = 1
    val time = 4
    RNG.setSeed(seed)

    val rec = Recurrent[Double]()
      .add(RnnCell[Double](inputSize, hiddenSize, Tanh()))
    val model = Sequential[Double]()
      .add(rec)

    val input = Tensor[Double](Array(batchSize, time, inputSize)).rand

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val state = rec.getState()

    state.toTensor[Double].map(output.asInstanceOf[Tensor[Double]].select(2, time), (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    rec.setState(state)
    model.forward(input)
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)], totalTime: Long = 0L): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })

    val allTime = if (totalTime > 0) {
      totalTime
    } else {
      forwardSum + backwardSum
    }
    println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}",
      s"whole time = ${allTime}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long, Long, Long, Double, Double)]
    var i = 0
    while (i < times.length) {
      val all = times(i)._2 + times(i)._3
      val rate = times(i)._3.toDouble/ times(i)._2
      val rateofAll = all.toDouble/allTime
      timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate, rateofAll))
      i += 1
    }
    val sortData = timeBuffer.sortBy(a => a._4)
    sortData.foreach(println)
  }

  "A Recurrent test " should "work good " in {
    val seed = 100

    val sequenceLen = 30
    val inputSize = 128
    val hiddenSize = 128
    val batchSize = 4

    val input = Tensor[Float](Array(batchSize, sequenceLen, inputSize)).randn()
    val labels = Tensor[Float](Array(batchSize, hiddenSize)).fill(1)
    val criterion = nn.MSECriterion[Float]()

    RNG.setSeed(seed)
    val model1 = nn.Sequential[Float]()
    model1.add(nn.Recurrent[Float]().add(RnnCell(inputSize, hiddenSize, Tanh[Float]())))
      .add(nn.Select(2, -1))

    RNG.setSeed(seed)
    val model2 = nn.Sequential[Float]()
    model2.add(nn.RecurrentNew[Float]().add(RnnCell[Float](inputSize, hiddenSize, Tanh[Float]())))
      .add(nn.Select(2, -1))

    val out1 = model1.forward(input)
    val grad1 = model1.backward(input, out1)
    val out2 = model2.forward(input)
    val grad2 = model2.backward(input, out2)

    // warm up
    for (i <- 1 to 100) {
      val out1 = model1.forward(input)
      val grad1 = model1.backward(input, out1)
    }
    for (i <- 1 to 100) {
      val out2 = model2.forward(input)
      val grad2 = model2.backward(input, out2)
    }

    //
    println("start run")
    val t2 = System.nanoTime()
    for (i <- 1 to 30) {
      val out2 = model2.forward(input)
      val grad2 = model2.backward(input, out2)
      //      val timeData = model2.getTimes()
      //      model2.resetTimes()
      //      getTopTimes(timeData)
      //      println("\n")
    }
    val end2 = System.nanoTime() -t2


    val t1 = System.nanoTime()
    for (i <- 1 to 30) {
      val out1 = model1.forward(input)
      val grad1 = model1.backward(input, out1)
//      val timeData = model1.getTimes()
//      model1.resetTimes()
//      getTopTimes(timeData)
//      println("\n")
    }
    val end1 = System.nanoTime() -t1

    println(s"end1 ${end1/1e9} end2 ${end2/1e9}")

    val w1 = model1.getParameters()
    val w2 = model2.getParameters()

    out1 should be(out2)
    grad1 should be(grad2)

    w1._1 should be (w2._1)
    w1._2 should be (w2._2)

    println("done")
  }

  "A Recurrent 11111 " should "work good " in {
    val seed = 100

    val sequenceLen = 30
    val hiddenSize = 128
    val batchSize = 4

    RNG.setSeed(100)
    val input = Tensor[Float](Array(batchSize, sequenceLen, hiddenSize)).fill(2.0f)

    input.select(2, 1).fill(0.01f)

    input.select(2, 4).fill(1.0f)

    input.select(2, 6).fill(10.0f)

    input.select(2, 8).fill(1.9999f)

    input.select(2, 11).fill(1.0f)

    input.select(2, 13).fill(1.0f)

    input.select(2, 20).fill(5.0f)

    val labels = Tensor[Float](Array(batchSize, sequenceLen)).fill(10.0f)
    RNG.setSeed(100)
    val criterion = nn.TimeDistributedCriterion[Float](CrossEntropyCriterion[Float](), true)

    val loss = criterion.forward(input, labels)

    println(loss)
    println("done")
  }

  "A Recurrent Module 1111111" should "perform correct gradient check" in {
    val hiddenSize = 128
    val inputSize = 128
    val seqLength = 30
    val bpttTruncate = 10
    val batchSize = 20
    val seed = 100

    RNG.setSeed(seed)
    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(LSTM[Double](inputSize, hiddenSize)))

    RNG.setSeed(seed)
    val model2 = SeqLSTM[Double](inputSize, hiddenSize)

    val (weights, grad) = model.getParameters()
    val (weights2, grad22) = model2.getParameters()

    weights2.copy(weights)
    grad22.copy(grad)

    val (t1, t2) = model2.getParameters()

    val input = Tensor[Double](Array(batchSize, seqLength, inputSize)).randn()
    val labels = Tensor[Double](Array(batchSize, seqLength, hiddenSize)).randn()

    for (i <- 1 to 20) {
      val out2 = model2.forward(input)
      val grad2 = model2.backward(input, labels)
    }

    for (i <- 1 to 20) {
      val out1 = model.forward(input)
      val grad1 = model.backward(input, labels)
    }

    for (i <- 1 to 30) {
      val start1 = System.nanoTime()
      val out1 = model.forward(input)
      val start2 = System.nanoTime()
      val grad1 = model.backward(input, labels)
      val start3 = System.nanoTime()
      val end1 = (start2 - start1)/1e9
      val end2 = (start3 - start2)/1e9

      println(s"lstm forward ${end1/1e9} backward ${end2/1e9}")
    }

    for (i <- 1 to 30) {
      val start1 = System.nanoTime()
      val out2 = model2.forward(input)
      val start2 = System.nanoTime()
      val grad2 = model2.backward(input, labels)
      val start3 = System.nanoTime()
      val end1 = (start2 - start1)/1e9
      val end2 = (start3 - start2)/1e9

      println(s"seqlstm  forward ${end1/1e9} backward ${end2/1e9}")
    }

    println("start")
    val start1 = System.nanoTime()
    for (i <- 1 to 20) {
      val out1 = model.forward(input)
      val grad1 = model.backward(input, labels)
    }

    val start2 = System.nanoTime()
    for (i <- 1 to 20) {
      val out2 = model2.forward(input)
      val grad2 = model2.backward(input, labels)
    }
    val end2 = System.nanoTime() - start2

    val time1 = model.getTimes()
    val time2 = model2.getTimes()
    val end1 = System.nanoTime() - start1
    println(s" end1 ${end1/1e9} end2 ${end2/1e9}")

    println("done")
  }

  "A Recurrent Module 2222222222" should "perform correct gradient check" in {
    val hiddenSize = 128
    val inputSize = 128
    val seqLength = 30
    val bpttTruncate = 10
    val batchSize = 20
    val seed = 100

    RNG.setSeed(seed)
    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(LSTM[Double](inputSize, hiddenSize)))

    RNG.setSeed(seed)
    val model2 = SeqLSTM[Double](inputSize, hiddenSize)

    val (weights, grad) = model.getParameters()
    val (weights2, grad22) = model2.getParameters()

    weights2.copy(weights)
    grad22.copy(grad)

    val (t1, t2) = model2.getParameters()

    val input = Tensor[Double](Array(batchSize, seqLength, inputSize)).randn()
    val labels = Tensor[Double](Array(batchSize, seqLength, hiddenSize)).randn()

    val out2 = model2.forward(input)
    val grad2 = model2.backward(input, labels)

    val out1 = model.forward(input)
    val grad1 = model.backward(input, labels)

    println("start")
    val start2 = System.nanoTime()
    for (i <- 1 to 200) {
      // val out2 = model2.forward(input)
      val grad2 = model2.backward(input, labels)
    }
    val end2 = System.nanoTime() - start2

    val start1 = System.nanoTime()
    for (i <- 1 to 200) {
      // val out1 = model.forward(input)
      val grad1 = model.backward(input, labels)
    }
    val time1 = model.getTimes()
    val time2 = model2.getTimes()
    val end1 = System.nanoTime() - start1
    println(s" end1 ${end1/1e9} end2 ${end2/1e9}")

    println("done")
  }

  "A Recurrent SeqLSTM Module" should "same with torch SeqLSTM" in {
    val hiddenSize = 3
    val inputSize = 1
    val outputSize = 3
    val bpttTruncate = 10
    val batchSize = 3
    val seed = 100

    RNG.setSeed(seed)
    val model2 = new SeqLSTM[Double](inputSize, hiddenSize).setName("lstm")


    val input = Tensor[Double](Array(batchSize, hiddenSize, inputSize)).rand()
    val labels = Tensor[Double](Array(batchSize, hiddenSize, hiddenSize)).rand()

    for (i <- 1 to 9) {
      val out2 = model2.forward(input)
      val grad2 = model2.backward(input, out2)
    }
    val out2 = model2.forward(input)
    val grad2 = model2.backward(input, out2)
    val allParams = model2.getParametersTable()
    val all2 = allParams.get("lstm").asInstanceOf[Option[Table]].get
    val weight = all2.get[Tensor[Double]]("weight").get
    val bias = all2.get[Tensor[Double]]("bias").get
    val gradWeight = all2.get[Tensor[Double]]("gradWeight").get
    val gradBias = all2.get[Tensor[Double]]("gradBias").get
    val gates = model2.gates
    val cell = model2.cell
    val buffer = model2.grad_a_buffer

    val code = "torch.manualSeed(" + seed + ")\n" +
      "require('torch')\n" +
      "require('nn')\n" +
      "require('rnn')\n" +
      "require('nngraph')" +
      "module = nn.SeqLSTM(1, 3)\n" +
      "module.batchfirst = true\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, output)\n" +
      "i = i + 1\n" +
      "end\n" +
      "bias = module.bias\n" +
      "weight = module.weight\n" +
      "gradBias = module.gradBias\n" +
      "gradWeight = module.gradWeight\n" +
      "rem = module._remember\n" +
      "c0 = module.c0\n" +
      "h0 = module.h0\n" +
      "gates = module.gates\n" +
      "cell = module.cell\n" +
      "grad_c0 = module.grad_c0\n" +
      "grad_h0 = module.grad_h0\n" +
      "buffer1 = module.buffer1\n" +
      "buffer2 = module.buffer2\n" +
      "grad_a_buffer = module.grad_a_buffer\n" +
      "grad_next_h = module.grad_next_h\n"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> labels),
      Array("output", "gradInput", "bias", "weight", "grad", "gradBias", "gradWeight", "rem",
        "c0", "h0", "gates", "cell", "grad_c0", "grad_h0", "buffer1", "buffer2", "grad_a_buffer",
        "grad_next_h"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]
    val luaGates = torchResult("gates").asInstanceOf[Tensor[Double]]
    val luagrad_a_buffer = torchResult("grad_a_buffer").asInstanceOf[Tensor[Double]]
    val grad_next_h = torchResult("grad_next_h").asInstanceOf[Tensor[Double]]
    val luaCell = torchResult("cell").asInstanceOf[Tensor[Double]]

    luaBias should be(bias)
    luaWeight should be(weight)
    //    luaGates should be(gates)
    luaOutput1 should be(out2)
    luaCell should be(cell)
    luagrad_a_buffer should be(buffer)
    luaGradBias should be(gradBias)
    luaGradWeight should be(gradWeight)
    luaOutput2 should be(grad2)

    println("done")
  }
}
