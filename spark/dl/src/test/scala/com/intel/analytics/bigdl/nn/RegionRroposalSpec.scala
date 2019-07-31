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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

class RegionRroposalSpec extends FlatSpec with Matchers {
  "RegionRroposal" should "be ok" in {
    val layer = new RegionRroposal(6,
      Array[Float](32, 64, 128, 256, 512),
      Array[Float](0.5f, 1.0f, 2.0f),
      Array[Float](4, 8, 16, 32, 64), 2000, 2000, 0.7f, 0, 2000)

    val images = Tensor[Float](T(T(20, 38)))

    val features = Tensor[Float](T(T(T(T(0.7668, 0.1659, 0.4393, 0.2243),
      T(0.8935, 0.0497, 0.1780, 0.3011),
      T(0.1893, 0.9186, 0.2131, 0.3957)),
      T(T(0.6017, 0.4234, 0.5224, 0.4175),
        T(0.0340, 0.9157, 0.3079, 0.6269),
        T(0.8277, 0.6594, 0.0887, 0.4890)),
      T(T(0.5887, 0.7340, 0.8497, 0.9112),
        T(0.4847, 0.9436, 0.3904, 0.2499),
        T(0.3206, 0.9753, 0.7582, 0.6688)),
      T(T(0.2651, 0.2336, 0.5057, 0.5688),
        T(0.0634, 0.8993, 0.2732, 0.3397),
        T(0.1879, 0.5534, 0.2682, 0.9556)),
      T(T(0.9761, 0.5934, 0.3124, 0.9431),
        T(0.8519, 0.9815, 0.1132, 0.4783),
        T(0.4436, 0.3847, 0.4521, 0.5569)),
      T(T(0.9952, 0.0015, 0.0813, 0.4907),
        T(0.2130, 0.4603, 0.1386, 0.0277),
        T(0.5662, 0.3503, 0.6555, 0.7667)))))

    val weight_conv = Tensor[Float](T(T(T(T(1.2685e-03, 1.3564e-02, 5.6322e-03),
      T(-1.0393e-03, -3.5746e-03, 3.9174e-03),
      T(-6.8009e-03, 2.4094e-03, 4.6981e-03)),
      T(T( 1.2426e-02, 5.4030e-03, -1.1454e-02),
        T(-1.4592e-02, -1.6281e-02, 3.8337e-03),
        T(-1.7180e-03, -3.1896e-02, 1.5914e-02)),
      T(T(-2.4669e-04, -8.4661e-03, 2.9301e-04),
        T(-5.7207e-03, -1.2546e-02, 4.8611e-04),
        T( 1.1705e-02, -5.4102e-03, -7.1156e-03)),
      T(T( 5.7526e-04, 6.2625e-03, -1.7736e-02),
        T(-2.2050e-03, 2.7467e-02, -1.7599e-02),
        T( 1.0230e-02, -1.1073e-03, -3.8986e-03)),
      T(T(-1.0300e-02, -1.5446e-02, 5.7298e-03),
        T(-2.0956e-02, -1.8055e-03, 2.3464e-03),
        T(-1.4774e-03, 5.8926e-03, 2.2533e-02)),
      T(T(-2.5548e-03, 1.6513e-03, -1.6292e-03),
        T(-8.0389e-03, -9.1740e-03, 8.9856e-03),
        T( 8.2623e-03, -3.6677e-03, -4.2506e-03))),
      T(T(T(-1.2455e-02, 1.1245e-02, -2.0157e-02),
        T( 9.9260e-03, -6.0842e-03, -1.3856e-02),
        T( 1.0412e-02, -8.0432e-03, -6.2443e-03)),
        T(T(-5.8823e-03, 1.6700e-02, -9.2747e-03),
          T(-9.7585e-03, 1.3312e-02, 9.0074e-03),
          T(-6.5847e-03, -9.3275e-03, -1.5749e-02)),
        T(T( 1.4861e-02, -1.4092e-02, 1.4330e-02),
          T( 3.8986e-03, -1.1516e-03, -2.3609e-03),
          T(-2.2235e-02, 7.8841e-04, 4.1560e-04)),
        T(T( 1.2813e-02, -8.2621e-03, 2.3098e-04),
          T( 1.9301e-02, 7.8028e-03, 3.1802e-03),
          T(-6.9918e-03, -3.9213e-03, 2.1955e-02)),
        T(T( 3.3116e-03, 1.4171e-03, -1.5268e-02),
          T( 2.5214e-03, 6.5413e-03, 2.1024e-02),
          T( 6.3311e-03, 1.9332e-02, -2.4634e-03)),
        T(T(-7.0092e-03, 6.3621e-03, -5.6589e-03),
          T( 1.0318e-02, -1.0371e-02, 1.3739e-03),
          T(-1.1312e-02, 6.4710e-03, -7.1830e-03))),
      T(T(T(-1.1984e-02, -8.8376e-03, 6.4301e-03),
        T( 7.2037e-04, -5.7234e-03, 1.6078e-02),
        T( 1.0007e-03, -1.0746e-02, -1.0924e-03)),
        T(T( 2.4635e-03, -9.9438e-03, -6.8856e-03),
          T( 1.2039e-02, -2.5186e-03, -1.9443e-02),
          T(-1.9203e-02, 1.1464e-02, 2.3850e-02)),
        T(T(-3.5508e-04, -3.1787e-03, 3.5779e-03),
          T(-1.7844e-02, -3.0524e-03, 8.5366e-03),
          T( 3.8534e-03, 1.2612e-02, 5.9866e-03)),
        T(T(-2.4725e-02, -5.4071e-04, -1.1862e-02),
          T( 7.3836e-03, -3.1864e-03, -5.1757e-03),
          T(-1.4699e-03, 5.1577e-03, 3.3928e-03)),
        T(T( 2.4955e-03, -9.5512e-03, 7.0652e-03),
          T( 1.2566e-02, -2.9903e-02, -3.2173e-04),
          T(-2.3036e-03, 1.2172e-03, 1.0538e-02)),
        T(T( 2.4320e-03, 8.3910e-03, 2.2082e-03),
          T(-1.3217e-02, 4.4108e-04, -3.4124e-03),
          T(-1.1553e-02, 4.9376e-03, 7.9121e-03))),
      T(T(T( 1.2293e-02, -3.9778e-03, 2.1020e-03),
        T( 8.3877e-03, 2.3666e-02, 6.8477e-03),
        T( 5.2052e-03, 1.4803e-02, -7.5135e-03)),
        T(T(-8.7030e-03, 5.8776e-03, -4.8942e-05),
          T( 2.0430e-02, 5.8311e-04, -3.6140e-03),
          T( 1.7116e-02, 8.4518e-03, -2.8076e-03)),
        T(T( 9.1432e-03, 4.6386e-03, -1.0463e-02),
          T( 6.0311e-03, 4.2746e-03, -3.4131e-03),
          T( 1.9404e-03, 7.9359e-03, -7.6828e-04)),
        T(T( 4.8792e-03, -2.5789e-02,  1.0007e-02),
          T( 2.1705e-04, -8.6712e-03, -4.5113e-03),
          T(-6.6698e-03,  2.7328e-04,  6.6046e-03)),

        T(T( 7.3924e-03,  7.1265e-03,  4.3357e-03),
          T( 3.9357e-04, -2.3774e-03,  6.4933e-03),
          T( 7.2543e-03, -4.8372e-03,  5.6666e-03)),

        T(T(-3.9601e-03,  1.3844e-02, -8.2588e-03),
          T(-1.6542e-03, -1.3295e-02,  3.8030e-03),
          T(-6.6701e-04,  6.8487e-03,  7.7284e-04))),


      T(T(T(-1.3936e-03, -4.7838e-03, -3.1820e-03),
        T( 2.2058e-03, -1.6855e-03,  1.8463e-02),
        T( 9.5022e-03, -3.3961e-03, -6.5992e-03)),

        T(T(-9.5200e-03, -4.0727e-03,  1.4081e-02),
          T( 1.2446e-03,  1.1088e-02,  1.7009e-03),
          T( 1.1670e-03, -7.9839e-03,  9.1257e-03)),

        T(T(-2.5381e-03,  6.8467e-03, -7.4647e-04),
          T( 5.9466e-04,  8.1772e-03,  2.8940e-03),
          T( 4.2105e-03, -1.3101e-02,  8.6801e-03)),

        T(T( 7.1093e-03,  9.3525e-03,  7.6763e-03),
          T(-2.8895e-03,  6.6717e-03,  1.1738e-03),
          T( 5.4419e-03, -2.8676e-04,  1.3919e-02)),

        T(T( 1.0932e-02, -2.3391e-02, -8.9627e-03),
          T(-6.2398e-03, -5.7453e-03, -5.7471e-03),
          T( 7.2978e-03, -2.2365e-03,  3.7101e-04)),

        T(T( 6.5447e-03, -2.5417e-03, -7.0376e-03),
          T(-1.1011e-03, -6.9527e-03, -2.4869e-02),
          T( 6.0163e-03,  5.7055e-03,  5.8137e-03))),


      T(T(T( 2.5749e-04,  5.5009e-03,  1.9151e-03),
        T( 9.8616e-03,  1.1613e-02, -1.7455e-03),
        T( 3.1561e-03, -1.8205e-03, -3.4044e-03)),

        T(T(-5.8910e-03,  3.6100e-03, -1.4282e-02),
          T( 9.2737e-03, -7.0391e-03,  3.8460e-03),
          T( 6.2735e-03,  6.5410e-03,  1.0932e-03)),

        T(T( 8.8084e-03,  1.5566e-02,  2.1806e-02),
          T( 1.7355e-02, -1.5105e-02,  7.6660e-04),
          T( 3.3541e-03, -5.3618e-03, -4.8840e-03)),

        T(T( 1.4804e-03,  4.5057e-03, -5.1785e-03),
          T(-5.5912e-03, -1.8077e-02,  5.0915e-03),
          T( 4.0559e-03,  3.3477e-03,  8.6055e-04)),

        T(T( 9.6151e-03, -2.7296e-03,  1.6761e-02),
          T(-6.7708e-03,  5.9753e-03, -5.5834e-03),
          T(-5.9345e-03,  2.2870e-02,  5.4827e-03)),

        T(T(-8.7740e-03,  1.4306e-02,  1.7519e-02),
          T(-1.0057e-04,  2.8130e-03, -1.4722e-02),
          T(-5.0060e-03,  8.9401e-04,  4.7907e-03)))))

    val weight_logits = Tensor[Float](T(T(T(T( 0.0013f)),

      T(T( 0.0136f)),

      T(T(-0.0002f)),

      T(T(-0.0085f)),

      T(T( 0.0003f)),

      T(T(-0.0057f))),
      T(T(T(-0.0125f)),

        T(T( 0.0005f)),

        T(T( 0.0028f)),

        T(T(-0.0215f)),

        T(T(-0.0071f)),

        T(T( 0.0006f))),
      T(T(T( 0.0063f)),

        T(T(-0.0177f)),

        T(T(-0.0022f)),

        T(T( 0.0275f)),

        T(T(-0.0105f)),

        T(T( 0.0112f)))))

    val weight_pred = Tensor[Float](T(T(T(T( 0.0013f)),
      T(T( 0.0136f)),
      T(T( 0.0056f)),

      T(T(-0.0010f)),

      T(T(-0.0036f)),

      T(T( 0.0039f))),
      T(T(T(-0.0068f)),

        T(T( 0.0024f)),

        T(T( 0.0047f)),

        T(T( 0.0124f)),

        T(T( 0.0054f)),

        T(T(-0.0115f))),


      T(T(T(-0.0146f)),

        T(T(-0.0163f)),

        T(T( 0.0038f)),

        T(T(-0.0017f)),

        T(T(-0.0319f)),

        T(T( 0.0159f))),


      T(T(T(-0.0002f)),

        T(T(-0.0085f)),

        T(T( 0.0003f)),

        T(T(-0.0057f)),

        T(T(-0.0125f)),

        T(T( 0.0005f))),


      T(T(T( 0.0117f)),

        T(T(-0.0054f)),

        T(T(-0.0071f)),

        T(T( 0.0006f)),

        T(T( 0.0063f)),

        T(T(-0.0177f))),


      T(T(T(-0.0022f)),

        T(T( 0.0275f)),

        T(T(-0.0176f)),

        T(T( 0.0102f)),

        T(T(-0.0011f)),

        T(T(-0.0039f))),
      T(T(T(-0.0103f)),

        T(T(-0.0154f)),

        T(T( 0.0057f)),

        T(T(-0.0210f)),

        T(T(-0.0018f)),

        T(T( 0.0023f))),
      T(T(T(-0.0015f)),

        T(T( 0.0059f)),

        T(T( 0.0225f)),

        T(T(-0.0026f)),

        T(T( 0.0017f)),

        T(T(-0.0016f))),


      T(T(T(-0.0080f)),

        T(T(-0.0092f)),

        T(T( 0.0090f)),

        T(T( 0.0083f)),

        T(T(-0.0037f)),

        T(T(-0.0043f))),
      T(T(T(-0.0125f)),

        T(T( 0.0112f)),

        T(T( 0.0044f)),

        T(T( 0.0142f)),

        T(T(-0.0043f)),

        T(T( 0.0030f))),
      T(T(T( 0.0266f)),

        T(T(-0.0028f)),

        T(T( 0.0017f)),

        T(T( 0.0100f)),

        T(T( 0.0022f)),

        T(T(-0.0036f))),
      T(T(T( 0.0081f)),

        T(T( 0.0002f)),

        T(T(-0.0084f)),

        T(T( 0.0124f)),

        T(T( 0.0151f)),

        T(T(-0.0060f)))))

    val paramsTable = layer.getParametersTable()

    for (i <- paramsTable.keySet) {
      val weight1 = paramsTable.get[Table](i).get.get[Tensor[Float]]("weight0").get
      weight1.copy(weight_conv)

      val weight2 = paramsTable.get[Table](i).get.get[Tensor[Float]]("weight2").get
      weight2.copy(weight_logits)

      val weight3 = paramsTable.get[Table](i).get.get[Tensor[Float]]("weight4").get
      weight3.copy(weight_pred)
    }

    layer.evaluate()
    val output = layer.forward(T(images, features))
    val outputExpected = Tensor[Float](T(
      T(0.0f, 0.0f,	20.999596f, 19.0f),
      T(0.0f, 0.0f,	12.995603f, 19.0f),
      T(0.0f, 0.0f,	37.0f, 19.0f),
      T(0.0f, 0.0f,	29.011127f, 13.003019f)
    ))

    output should be(outputExpected)
  }
}
