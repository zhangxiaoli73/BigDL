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
package com.intel.analytics.bigdl.nn.rnn

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class AttentionSpec  extends FlatSpec with Matchers {
  "attention layer" should "same with tf" in {
    val attention = new AttentionLayer[Float](8, 4, 0.1f)

    val arrX = Tensor[Float](T(T(T(0.2390374 , 0.92039955, 0.05051243, 0.49574447, 0.8355223 ,
      0.02647042, 0.08811307, 0.4566604 ),
      T(0.76883924, 0.7376363 , 0.78504944, 0.31202638, 0.15186465,
        0.20276117, 0.3083856 , 0.5401472 ),
      T(0.45097053, 0.2531916 , 0.29826486, 0.9727912 , 0.19891858,
        0.7108258 , 0.80928993, 0.8320373 )),

      T(T(0.03928626, 0.42656052, 0.37600386, 0.44390714, 0.21000767,
        0.46763003, 0.6666417 , 0.9928988 ),
        T(0.12868619, 0.8806071 , 0.7419156 , 0.5514581 , 0.9784763 ,
          0.80584776, 0.23819816, 0.43904757),
        T(0.20762491, 0.36509156, 0.89200187, 0.01050866, 0.8644662 ,
          0.3498454 , 0.17683744, 0.538926  ))))

    val arrBias = Tensor[Float](T(T(T(T(0.3869294 , 0.36002624, 0.9768796 ),
      T(0.08651638, 0.5038012 , 0.3585019 ),
      T(0.20932388, 0.02714252, 0.4244014 )),

      T(T(0.5076815 , 0.3446772 , 0.48250294),
        T(0.01125383, 0.8801354 , 0.22752082),
        T(0.11322451, 0.26014864, 0.6512549 )),

      T(T(0.52905834, 0.537508  , 0.9278991 ),
        T(0.4385563 , 0.07387316, 0.50058925),
        T(0.43982077, 0.87719023, 0.5540434 )),

      T(T(0.71598864, 0.37673223, 0.37794995),
        T(0.02403736, 0.49887884, 0.7218306 ),
        T(0.61134934, 0.2513523 , 0.7746686 ))),

      T(T(T(0.7968825 , 0.22291362, 0.575938  ),
        T(0.5907738 , 0.16110945, 0.26993048),
        T(0.05779219, 0.13434005, 0.49843323)),

        T(T(0.3045802 , 0.11464524, 0.35308015),
          T(0.9469961 , 0.68015385, 0.5188323 ),
          T(0.12080741, 0.03275084, 0.22498906)),

        T(T(0.8535242 , 0.35764086, 0.8901398 ),
          T(0.4488008 , 0.4906304 , 0.15883219),
          T(0.40362644, 0.6043719 , 0.69416916)),

        T(T(0.7486639 , 0.6879139 , 0.21003795),
          T(0.31096447, 0.50465226, 0.05503154),
          T(0.6366972 , 0.04477406, 0.35868895)))))


    val x = arrX // Tensor[Float](2, 3, 8).fill(1.0f)
    val y = x.clone()
    val bias = arrBias // Tensor[Float](2, 4, 3, 3).fill(2.0f)

    val output = attention.updateOutput(T(x, y, bias))

    val gradOutput = Tensor[Float](2, 3, 8).rand()
    val gradInput = attention.updateGradInput(T(x, y, bias), gradOutput.asInstanceOf[Activity])
    // output should be(outputTemp)
    println("done")
  }
}
