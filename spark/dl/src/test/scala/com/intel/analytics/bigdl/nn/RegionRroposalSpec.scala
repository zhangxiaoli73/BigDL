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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class RegionRroposalSpec extends FlatSpec with Matchers {
  "RegionRroposal" should "be ok" in {
    val layer = new RegionRroposal(6,
      Array[Float](32, 64, 128, 256, 512),
      Array[Float](0.5f, 1.0f, 2.0f),
      Array[Float](4, 8, 16, 32, 64), 2000, 2000, 0.7f, 0, 2000)

    val images = Tensor[Float](T(20, 38))

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
        T(T( 4.8792e-03, -2.5789e-02, 1.0007e-02),
          T( 2.1705e-04, -8.6712e-03, -4.5113e-03),
          T(-6.6698e-03, 2.7328e-04, 6.6046e-03)),
        T(T( 7.3924e-03, 7.1265e-03, 4.3357e-03),
          T( 3.9357e-04, -2.3774e-03, 6.4933e-03),
          T( 7.2543e-03, -4.8372e-03, 5.6666e-03)),
        T(T(-3.9601e-03, 1.3844e-02, -8.2588e-03),
          T(-1.6542e-03, -1.3295e-02, 3.8030e-03),
          T(-6.6701e-04, 6.8487e-03, 7.7284e-04))),
      T(T(T(-1.3936e-03, -4.7838e-03, -3.1820e-03),
        T( 2.2058e-03, -1.6855e-03, 1.8463e-02),
        T( 9.5022e-03, -3.3961e-03, -6.5992e-03)),
        T(T(-9.5200e-03, -4.0727e-03, 1.4081e-02),
          T( 1.2446e-03, 1.1088e-02, 1.7009e-03),
          T( 1.1670e-03, -7.9839e-03, 9.1257e-03)),
        T(T(-2.5381e-03, 6.8467e-03, -7.4647e-04),
          T( 5.9466e-04, 8.1772e-03, 2.8940e-03),
          T( 4.2105e-03, -1.3101e-02, 8.6801e-03)),
        T(T( 7.1093e-03, 9.3525e-03, 7.6763e-03),
          T(-2.8895e-03, 6.6717e-03, 1.1738e-03),
          T( 5.4419e-03, -2.8676e-04, 1.3919e-02)),
        T(T( 1.0932e-02, -2.3391e-02, -8.9627e-03),
          T(-6.2398e-03, -5.7453e-03, -5.7471e-03),
          T( 7.2978e-03, -2.2365e-03, 3.7101e-04)),
        T(T( 6.5447e-03, -2.5417e-03, -7.0376e-03),
          T(-1.1011e-03, -6.9527e-03, -2.4869e-02),
          T( 6.0163e-03, 5.7055e-03, 5.8137e-03))),
      T(T(T( 2.5749e-04, 5.5009e-03, 1.9151e-03),
        T( 9.8616e-03, 1.1613e-02, -1.7455e-03),
        T( 3.1561e-03, -1.8205e-03, -3.4044e-03)),
        T(T(-5.8910e-03, 3.6100e-03, -1.4282e-02),
          T( 9.2737e-03, -7.0391e-03, 3.8460e-03),
          T( 6.2735e-03, 6.5410e-03, 1.0932e-03)),
        T(T( 8.8084e-03, 1.5566e-02, 2.1806e-02),
          T( 1.7355e-02, -1.5105e-02, 7.6660e-04),
          T( 3.3541e-03, -5.3618e-03, -4.8840e-03)),
        T(T( 1.4804e-03, 4.5057e-03, -5.1785e-03),
          T(-5.5912e-03, -1.8077e-02, 5.0915e-03),
          T( 4.0559e-03, 3.3477e-03, 8.6055e-04)),
        T(T( 9.6151e-03, -2.7296e-03, 1.6761e-02),
          T(-6.7708e-03, 5.9753e-03, -5.5834e-03),
          T(-5.9345e-03, 2.2870e-02, 5.4827e-03)),
        T(T(-8.7740e-03, 1.4306e-02, 1.7519e-02),
          T(-1.0057e-04, 2.8130e-03, -1.4722e-02),
          T(-5.0060e-03, 8.9401e-04, 4.7907e-03)))))

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
      val weight = paramsTable.get[Table](i).get.get[Tensor[Float]]("weight").get
      if (i.toString contains "_cls_logits") {
        weight.copy(weight_logits)
      } else if (i.toString contains "_bbox_pred") {
        weight.copy(weight_pred)
      } else {
        weight.copy(weight_conv)
      }
    }

    layer.evaluate()
    val output = layer.forward(T(images, T(features)))
    val outputExpected = Tensor[Float](
      T(T(0.0f, 0.0f, 20.999596f, 19.0f),
      T(0.0f, 0.0f, 12.995603f, 19.0f),
      T(0.0f, 0.0f, 37.0f, 19.0f),
      T(0.0f, 0.0f, 29.011127f, 13.003019f)
    ))

    output should be(outputExpected)
  }

  "RegionRroposal with multi features" should "be ok" in {
    val layer = new RegionRroposal(6,
      Array[Float](32, 64, 128, 256, 512),
      Array[Float](0.5f, 1.0f, 2.0f),
      Array[Float](4, 8, 16, 32, 64), 2000, 2000, 0.7f, 0, 2000)

    val images = Tensor[Float](T(20, 38))


    val features1 = Tensor[Float](T(T(T(
      T(0.7668, 0.1659, 0.4393, 0.2243),
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

    val features3 = Tensor[Float](T(T(T(
      T(0.9336, 0.2557, 0.1506, 0.7856)),

      T(T(0.4152, 0.5809, 0.1088, 0.7065)),

      T(T(0.0105, 0.4602, 0.2945, 0.0475)),

      T(T(0.6401, 0.3784, 0.5887, 0.0720)),

      T(T(0.9140, 0.0085, 0.2174, 0.1890)),

      T(T(0.0911, 0.6344, 0.3142, 0.7052)))))

    val features2 = Tensor[Float](T(T(T(T(0.2269, 0.7555),
      T(0.6458, 0.3673),
      T(0.1770, 0.2966),
      T(0.9925, 0.2103),
      T(0.1292, 0.1719)),

      T(T(0.9127, 0.6818),
        T(0.1953, 0.9991),
        T(0.1133, 0.0135),
        T(0.1450, 0.7819),
        T(0.3134, 0.2983)),

      T(T(0.3436, 0.2028),
        T(0.9792, 0.4947),
        T(0.3617, 0.9687),
        T(0.0359, 0.3041),
        T(0.9867, 0.1290)),

      T(T(0.6887, 0.1637),
        T(0.0899, 0.3139),
        T(0.1219, 0.3516),
        T(0.2316, 0.2847),
        T(0.3520, 0.2828)),

      T(T(0.2420, 0.4928),
        T(0.5772, 0.3771),
        T(0.2440, 0.8994),
        T(0.1041, 0.9193),
        T(0.6201, 0.3658)),

      T(T(0.0623, 0.5967),
        T(0.0829, 0.8185),
        T(0.4964, 0.0589),
        T(0.9840, 0.5836),
        T(0.6737, 0.4738)))))

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
        T(T( 4.8792e-03, -2.5789e-02, 1.0007e-02),
          T( 2.1705e-04, -8.6712e-03, -4.5113e-03),
          T(-6.6698e-03, 2.7328e-04, 6.6046e-03)),
        T(T( 7.3924e-03, 7.1265e-03, 4.3357e-03),
          T( 3.9357e-04, -2.3774e-03, 6.4933e-03),
          T( 7.2543e-03, -4.8372e-03, 5.6666e-03)),
        T(T(-3.9601e-03, 1.3844e-02, -8.2588e-03),
          T(-1.6542e-03, -1.3295e-02, 3.8030e-03),
          T(-6.6701e-04, 6.8487e-03, 7.7284e-04))),
      T(T(T(-1.3936e-03, -4.7838e-03, -3.1820e-03),
        T( 2.2058e-03, -1.6855e-03, 1.8463e-02),
        T( 9.5022e-03, -3.3961e-03, -6.5992e-03)),
        T(T(-9.5200e-03, -4.0727e-03, 1.4081e-02),
          T( 1.2446e-03, 1.1088e-02, 1.7009e-03),
          T( 1.1670e-03, -7.9839e-03, 9.1257e-03)),
        T(T(-2.5381e-03, 6.8467e-03, -7.4647e-04),
          T( 5.9466e-04, 8.1772e-03, 2.8940e-03),
          T( 4.2105e-03, -1.3101e-02, 8.6801e-03)),
        T(T( 7.1093e-03, 9.3525e-03, 7.6763e-03),
          T(-2.8895e-03, 6.6717e-03, 1.1738e-03),
          T( 5.4419e-03, -2.8676e-04, 1.3919e-02)),
        T(T( 1.0932e-02, -2.3391e-02, -8.9627e-03),
          T(-6.2398e-03, -5.7453e-03, -5.7471e-03),
          T( 7.2978e-03, -2.2365e-03, 3.7101e-04)),
        T(T( 6.5447e-03, -2.5417e-03, -7.0376e-03),
          T(-1.1011e-03, -6.9527e-03, -2.4869e-02),
          T( 6.0163e-03, 5.7055e-03, 5.8137e-03))),
      T(T(T( 2.5749e-04, 5.5009e-03, 1.9151e-03),
        T( 9.8616e-03, 1.1613e-02, -1.7455e-03),
        T( 3.1561e-03, -1.8205e-03, -3.4044e-03)),
        T(T(-5.8910e-03, 3.6100e-03, -1.4282e-02),
          T( 9.2737e-03, -7.0391e-03, 3.8460e-03),
          T( 6.2735e-03, 6.5410e-03, 1.0932e-03)),
        T(T( 8.8084e-03, 1.5566e-02, 2.1806e-02),
          T( 1.7355e-02, -1.5105e-02, 7.6660e-04),
          T( 3.3541e-03, -5.3618e-03, -4.8840e-03)),
        T(T( 1.4804e-03, 4.5057e-03, -5.1785e-03),
          T(-5.5912e-03, -1.8077e-02, 5.0915e-03),
          T( 4.0559e-03, 3.3477e-03, 8.6055e-04)),
        T(T( 9.6151e-03, -2.7296e-03, 1.6761e-02),
          T(-6.7708e-03, 5.9753e-03, -5.5834e-03),
          T(-5.9345e-03, 2.2870e-02, 5.4827e-03)),
        T(T(-8.7740e-03, 1.4306e-02, 1.7519e-02),
          T(-1.0057e-04, 2.8130e-03, -1.4722e-02),
          T(-5.0060e-03, 8.9401e-04, 4.7907e-03)))))

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
      val weight = paramsTable.get[Table](i).get.get[Tensor[Float]]("weight").get
      if (i.toString contains "_cls_logits") {
        weight.copy(weight_logits)
      } else if (i.toString contains "_bbox_pred") {
        weight.copy(weight_pred)
      } else {
        weight.copy(weight_conv)
      }
    }

    layer.evaluate()
    val output = layer.forward(T(images, T(features1, features2, features3)))
    val outputExpected = Tensor[Float](T(
      T( 0.0000,  0.0000, 35.0363, 19.0000),
      T( 0.0000,  0.0000, 20.9997, 19.0000),
      T( 0.0000,  0.0000, 12.9955, 19.0000),
      T( 0.0000,  0.0000, 37.0000, 19.0000),
      T( 0.0000,  0.0000, 37.0000, 19.0000),
      T(11.9914,  0.0000, 37.0000, 19.0000),
      T( 0.0000,  0.0000, 29.0113, 13.0032),
      T( 0.0000, 11.9920, 37.0000, 19.0000)))

      output should be(outputExpected)
  }

  "RPNPostProcessor" should "be ok" in {
    val anchors = Tensor[Float](
      T(T(-22, -10,  25,  13),
        T(-14, -14,  17,  17),
        T(-10, -22,  13,  25),
        T(-18, -10,  29,  13),
        T(-10, -14,  21,  17),
        T( -6, -22,  17,  25),
        T(-14, -10,  33,  13),
        T( -6, -14,  25,  17),
        T( -2, -22,  21,  25),
        T(-10, -10,  37,  13),
        T( -2, -14,  29,  17),
        T(  2, -22,  25,  25),
        T( -6, -10,  41,  13),
        T(  2, -14,  33,  17),
        T(  6, -22,  29,  25),
        T( -2, -10,  45,  13),
        T(  6, -14,  37,  17),
        T( 10, -22,  33,  25),
        T(  2, -10,  49,  13),
        T( 10, -14,  41,  17),
        T( 14, -22,  37,  25),
        T(-22,  -6,  25,  17),
        T(-14, -10,  17,  21),
        T(-10, -18,  13,  29),
        T(-18,  -6,  29,  17),
        T(-10, -10,  21,  21),
        T( -6, -18,  17,  29),
        T(-14,  -6,  33,  17),
        T( -6, -10,  25,  21),
        T( -2, -18,  21,  29),
        T(-10,  -6,  37,  17),
        T( -2, -10,  29,  21),
        T(  2, -18,  25,  29),
        T( -6,  -6,  41,  17),
        T(  2, -10,  33,  21),
        T(  6, -18,  29,  29),
        T( -2,  -6,  45,  17),
        T(  6, -10,  37,  21),
        T( 10, -18,  33,  29),
        T(  2,  -6,  49,  17),
        T( 10, -10,  41,  21),
        T( 14, -18,  37,  29),
        T(-22,  -2,  25,  21),
        T(-14,  -6,  17,  25),
        T(-10, -14,  13,  33),
        T(-18,  -2,  29,  21),
        T(-10,  -6,  21,  25),
        T( -6, -14,  17,  33),
        T(-14,  -2,  33,  21),
        T( -6,  -6,  25,  25),
        T( -2, -14,  21,  33),
        T(-10,  -2,  37,  21),
        T( -2,  -6,  29,  25),
        T(  2, -14,  25,  33),
        T( -6,  -2,  41,  21),
        T(  2,  -6,  33,  25),
        T(  6, -14,  29,  33),
        T( -2,  -2,  45,  21),
        T(  6,  -6,  37,  25),
        T( 10, -14,  33,  33),
        T(  2,  -2,  49,  21),
        T( 10,  -6,  41,  25),
        T( 14, -14,  37,  33)))
    val box_regression = Tensor[Float](
      T(T(T(T(-1.6730e-02, -2.5040e-02, -3.8669e-02, -2.5333e-02, -1.4004e-02,
        -2.5377e-02, -1.2593e-02),
        T(-3.6522e-02, -1.0507e-02, -2.6155e-02, -3.6207e-02, -2.4963e-02,
          -2.1895e-02, -1.5993e-02),
        T(-1.6325e-02, -2.7535e-02, -1.6704e-02, -1.4899e-02, -1.1344e-02,
          -3.0802e-03, -1.2747e-02)),
        T(T(-7.5157e-03, -2.8978e-02, -2.8847e-02, -4.5879e-02, -3.0130e-02,
          -3.3889e-02, -5.1871e-02),
          T(-2.1900e-02, -2.2046e-02, -2.7110e-02, -3.2612e-02, -2.8986e-02,
            -6.6867e-02, -7.1081e-02),
          T( 3.0462e-03, -2.0255e-02, -3.9770e-02, -3.5203e-02, -4.7388e-02,
            -2.4220e-02, -4.6222e-02)),
        T(T( 5.7844e-04, -1.6412e-04,  9.7524e-03, -6.9274e-03,  1.7444e-06,
          5.4107e-03, -2.1182e-02),
          T(-1.5361e-02,  2.2865e-02,  1.7374e-02,  2.8522e-03,  3.3781e-02,
            1.0332e-02,  1.0356e-02),
          T( 3.3926e-03,  3.6011e-02,  1.8886e-02,  2.5415e-02,  2.0812e-02,
            2.1618e-02,  2.0776e-02)),
        T(T( 5.3066e-02,  5.4734e-02,  5.1326e-02,  3.5983e-02,  5.5721e-02,
          5.8108e-02,  3.7270e-02),
          T( 7.3613e-02,  5.4528e-02,  6.9086e-02,  5.8593e-02,  3.3255e-02,
            7.0331e-02,  3.9792e-02),
          T( 4.0440e-02,  4.5344e-02,  3.0102e-02,  3.9423e-02,  3.7462e-02,
            1.9178e-02,  3.4250e-02)),
        T(T( 9.3921e-03, -6.3640e-03,  6.6344e-03, -2.9477e-02,  2.8380e-03,
          2.4094e-04, -3.8125e-02),
          T( 1.3277e-02,  3.2003e-02,  9.2812e-03,  3.1793e-02,  3.5682e-02,
            5.4143e-03, -2.7538e-02),
          T(-1.4505e-02,  4.2906e-03, -5.5038e-03,  1.1895e-02, -8.9942e-03,
            9.1047e-03, -5.2846e-03)),
        T(T(-2.4140e-02, -4.9850e-02, -8.1354e-03, -4.0075e-02, -2.3858e-02,
          -1.0505e-02, -1.8872e-03),
          T(-5.3244e-02, -5.0973e-02, -5.3102e-02, -3.2843e-02, -4.9433e-02,
            -2.6899e-02, -2.1426e-02),
          T(-3.8070e-02, -3.4148e-02, -2.2365e-02, -1.0786e-02, -2.1428e-03,
            -2.9661e-02,  6.5642e-03)),
        T(T( 7.1718e-03, -1.8317e-02, -1.9746e-02,  3.5586e-04,  5.8551e-04,
          1.3969e-02, -2.5201e-03),
          T(-1.3888e-02, -9.6641e-03, -3.8934e-02, -2.8148e-02, -2.5934e-02,
            -1.8294e-02, -2.0061e-02),
          T( 1.0523e-02,  2.6551e-02, -2.9795e-02, -9.7123e-03, -1.4083e-03,
            -2.3482e-02, -1.5405e-02)),
        T(T( 2.5275e-02,  1.6022e-02,  2.1474e-02,  2.3938e-02,  1.6918e-02,
          2.9566e-02,  1.6430e-02),
          T(-8.9619e-03, -1.5747e-02,  2.2626e-02,  9.3860e-03, -2.7444e-03,
            1.0630e-02,  4.0585e-03),
          T(-2.6552e-02, -4.6460e-02, -1.1829e-02, -5.0394e-02, -2.1685e-02,
            -1.0684e-02, -3.7224e-02)),
        T(T( 8.2827e-03,  1.7244e-02,  2.7117e-02,  9.7096e-05,  3.1359e-02,
          4.6453e-03,  9.5188e-03),
          T( 4.0039e-02,  4.7410e-02,  9.9494e-03,  2.4956e-02,  2.7872e-02,
            2.4829e-02,  1.5199e-02),
          T( 2.1342e-02,  3.1655e-02,  2.1581e-02,  2.5497e-02,  5.2575e-02,
            2.4982e-02,  2.5912e-02)),
        T(T(-3.8185e-02, -3.9303e-02, -4.1358e-02, -4.0111e-02, -1.3078e-02,
          -2.2576e-02, -2.8542e-02),
          T(-3.6325e-02, -4.7150e-02, -1.7211e-02, -1.9650e-02,  5.6505e-04,
            -4.6043e-03, -4.4149e-02),
          T( 1.2474e-03, -2.1102e-02, -2.4141e-02,  9.8825e-03, -2.2259e-02,
            -1.1524e-02, -1.6652e-04)),
        T(T(-1.6188e-02, -2.3977e-02,  1.8660e-02, -1.5378e-02, -2.7290e-02,
          -2.5314e-02, -1.1265e-02),
          T(-2.8503e-02, -1.7718e-02, -5.1043e-03, -3.6894e-02, -1.6136e-02,
            -3.3021e-02, -1.9824e-02),
          T(-2.8551e-02, -3.7279e-02, -2.3878e-02, -2.9096e-02, -2.2290e-02,
            -2.6733e-02, -2.2998e-02)),
        T(T( 5.0010e-03, -8.0676e-03, -1.4430e-02, -1.5388e-02,  1.0738e-02,
          3.8478e-03,  2.1696e-03),
          T(-2.3630e-03, -4.0806e-02, -2.7923e-02, -1.1444e-02,  3.1605e-03,
            -1.7883e-02, -3.3700e-02),
          T( 5.6951e-03,  1.8676e-02, -2.4579e-03,  1.0234e-02,  3.3008e-03,
            3.0289e-03,  3.3703e-02))))
    )
    val objectness = Tensor[Float](T(T(T(
      T(-0.0429, -0.0315, -0.0317, -0.0458, -0.0145, -0.0326, -0.0305),
      T(-0.0361, -0.0716, -0.0414, -0.0237, -0.0399, -0.0334, -0.0345),
      T(-0.0168, -0.0163, -0.0441, -0.0193, -0.0388, -0.0227, -0.0345)),
      T(T( 0.0194, -0.0012,  0.0251, -0.0154, -0.0265, -0.0014,  0.0094),
        T( 0.0443,  0.0278,  0.0358,  0.0061,  0.0576,  0.0287,  0.0263),
        T(-0.0037, -0.0024,  0.0217,  0.0264,  0.0165,  0.0058,  0.0382)),
      T(T(-0.0011, -0.0058, -0.0089, -0.0017, -0.0266, -0.0007, -0.0156),
        T( 0.0087,  0.0164, -0.0103,  0.0014, -0.0262,  0.0151,  0.0157),
        T(-0.0223,  0.0009, -0.0051, -0.0074, -0.0148, -0.0156, -0.0043)))))

    val preNmsTopN: Int = 2000
    val postNmsTopN: Int = 2000
    val rpnPreNmsTopNTrain: Int = 2000

    val proposal = new ProposalPostProcessor(2000, 2000, 0.7f, 0, 2000)
    val output = proposal.forward(T(anchors, objectness, box_regression, Tensor[Float](T(20, 38))))

    val expectOutput = Tensor[Float](
      T(T(3.5029516,	0.0,	33.70933,	19.0),
        T(0.0,	0.0,	17.197811,	19.0),
        T(0.0,	0.0,	24.902605,	17.08425),
        T(14.575309,	0.0,	37.0,	19.0),
        T(0.0,	0.0,	37.0,	12.965991))
    )

    output[Tensor[Float]](1) should be(expectOutput)
    println("done")
  }

  "AnchorGenerate" should "be ok" in {
    val layer = new RegionRroposal(6,
      Array[Float](32, 64, 128, 256, 512),
      Array[Float](0.5f, 1.0f, 2.0f),
      Array[Float](4, 8, 16, 32, 64), 2000, 2000, 0.7f, 0, 2000)

    val input = Tensor[Float](T(T(T(T(0.7668, 0.1659, 0.4393, 0.2243),
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

    val expectedOutput = Tensor[Float](T(T(-22, -10, 25, 13),
      T(-14, -14, 17, 17),
      T(-10, -22, 13, 25),
      T(-18, -10, 29, 13),
      T(-10, -14, 21, 17),
      T( -6, -22, 17, 25),
      T(-14, -10, 33, 13),
      T( -6, -14, 25, 17),
      T( -2, -22, 21, 25),
      T(-10, -10, 37, 13),
      T( -2, -14, 29, 17),
      T(  2, -22, 25, 25),
      T(-22, -6, 25, 17),
      T(-14, -10, 17, 21),
      T(-10, -18, 13, 29),
      T(-18, -6, 29, 17),
      T(-10, -10, 21, 21),
      T( -6, -18, 17, 29),
      T(-14, -6, 33, 17),
      T( -6, -10, 25, 21),
      T( -2, -18, 21, 29),
      T(-10, -6, 37, 17),
      T( -2, -10, 29, 21),
      T(  2, -18, 25, 29),
      T(-22, -2, 25, 21),
      T(-14, -6, 17, 25),
      T(-10, -14, 13, 33),
      T(-18, -2, 29, 21),
      T(-10, -6, 21, 25),
      T( -6, -14, 17, 33),
      T(-14, -2, 33, 21),
      T( -6, -6, 25, 25),
      T( -2, -14, 21, 33),
      T(-10, -2, 37, 21),
      T( -2, -6, 29, 25),
      T(  2, -14, 25, 33)))

    val output = layer.anchorGenerater(T(input))

    output.apply[Tensor[Float]](1) should be(expectedOutput)
  }
}

class RegionRroposalSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = new RegionRroposal(6,
      Array[Float](32, 64, 128, 256, 512),
      Array[Float](0.5f, 1.0f, 2.0f),
      Array[Float](4, 8, 16, 32, 64), 2000, 2000, 0.7f, 0, 2000).setName("RegionRroposal")

    val features = Tensor[Float](1, 6, 3, 4).rand()
    val imgInfo = Tensor[Float](T(20, 38))
    runSerializationTest(layer, T(features, imgInfo))
  }
}
