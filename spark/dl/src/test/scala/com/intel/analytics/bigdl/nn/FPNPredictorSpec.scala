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
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class FPNPredictorSpec extends FlatSpec with Matchers {
  "FPNPredictor" should "be ok" in {
    val num_class: Int = 81
    val in_channels: Int = 1024
    val num_bbox_reg_classes: Int = 81

  val input = Tensor[Float](T(
    T(2.2933e-01, 4.7398e-01, 0.0000e+00, 0.0000e+00, 3.0377e-01, 0.0000e+00,
    1.5809e-01, 0.0000e+00, 3.4213e-03, 2.0427e-02, 0.0000e+00, 0.0000e+00,
    1.8175e-01, 3.0648e-01, 3.0486e-01, 4.8535e-01, 1.0736e-01, 0.0000e+00,
    0.0000e+00, 1.1959e-01, 4.1052e-01, 0.0000e+00, 0.0000e+00, 2.6293e-01,
    4.3437e-01, 7.2744e-01, 2.8527e-01, 0.0000e+00, 5.7330e-01, 3.2637e-02,
    3.6768e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.2849e-01,
    0.0000e+00, 7.6325e-01, 5.8331e-01, 1.9046e-01, 0.0000e+00, 6.1702e-01,
    0.0000e+00, 0.0000e+00, 4.2936e-01, 2.5023e-01, 1.5075e-02, 8.7771e-02,
    0.0000e+00, 5.7307e-01, 0.0000e+00, 8.0079e-01, 0.0000e+00, 0.0000e+00,
    2.6381e-03, 0.0000e+00, 1.1355e-01, 3.3714e-01, 1.6419e-01, 0.0000e+00,
    1.2210e-01, 0.0000e+00, 3.1064e-01, 2.6534e-01, 0.0000e+00, 9.3856e-01,
    0.0000e+00, 2.0916e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8734e-01,
    0.0000e+00, 4.2838e-01, 4.2797e-01, 0.0000e+00, 0.0000e+00, 1.0801e-01,
    0.0000e+00, 7.2708e-02, 6.4211e-02, 2.4386e-01, 4.8236e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.1248e-01, 0.0000e+00,
    3.6260e-01, 0.0000e+00, 2.6733e-01, 4.3891e-01, 0.0000e+00, 9.4399e-01,
    4.3039e-01, 1.9963e-01, 5.5950e-01, 0.0000e+00, 5.7901e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 3.4324e-02, 0.0000e+00, 5.7276e-02, 5.4953e-01,
    1.1287e-01, 7.1539e-02, 6.8938e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    7.2902e-02, 1.1641e-01, 4.0944e-02, 1.0786e-01, 7.0635e-01, 0.0000e+00,
    6.2662e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    4.1090e-01, 0.0000e+00, 0.0000e+00, 3.8734e-01, 0.0000e+00, 9.4146e-02,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 3.6706e-01, 0.0000e+00, 0.0000e+00,
    9.5533e-01, 2.9218e-01, 7.2061e-01, 0.0000e+00, 3.2076e-01, 0.0000e+00,
    1.9556e-01, 0.0000e+00, 7.7142e-01, 2.0086e-01, 3.6382e-02, 0.0000e+00,
    2.4287e-01, 5.7229e-01, 0.0000e+00, 0.0000e+00, 2.8146e-01, 0.0000e+00,
    2.1619e-01, 0.0000e+00, 5.3419e-01, 0.0000e+00, 3.7420e-01, 0.0000e+00,
    3.9476e-03, 0.0000e+00, 0.0000e+00, 2.9974e-01, 2.0722e-01, 1.1056e-01,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.5141e-01, 0.0000e+00,
    0.0000e+00, 8.1091e-01, 6.0005e-01, 5.3662e-03, 5.5893e-02, 0.0000e+00,
    9.1367e-01, 3.9175e-01, 4.4933e-01, 3.6535e-01, 3.2467e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    4.4286e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.5753e-01, 0.0000e+00,
    0.0000e+00, 4.3492e-02, 0.0000e+00, 3.9084e-01, 4.5749e-01, 0.0000e+00,
    8.8619e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.8256e-01, 4.3740e-01,
    5.4134e-01, 8.1222e-01, 1.1236e+00, 3.8259e-01, 0.0000e+00, 4.0677e-01,
    0.0000e+00, 6.1217e-01, 0.0000e+00, 0.0000e+00, 9.0240e-02, 0.0000e+00,
    0.0000e+00, 3.7207e-01, 3.4780e-01, 0.0000e+00, 8.2428e-02, 0.0000e+00,
    0.0000e+00, 6.4101e-02, 6.2217e-01, 0.0000e+00, 0.0000e+00, 1.4846e-01,
    1.4339e-01, 0.0000e+00, 0.0000e+00, 1.4410e-01, 9.8182e-01, 6.2625e-01,
    0.0000e+00, 5.8311e-01, 3.7540e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    2.7168e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.5780e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 2.5917e-01, 3.9157e-01, 2.2357e-01,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0621e-01, 0.0000e+00, 0.0000e+00,
    1.5849e-01, 0.0000e+00, 2.8027e-02, 1.3950e-01, 0.0000e+00, 4.5800e-01,
    4.8431e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 3.2327e-01, 3.9075e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 2.7243e-01, 6.9952e-01, 3.3709e-01, 7.4156e-02,
    0.0000e+00, 0.0000e+00, 4.3161e-01, 0.0000e+00, 5.5968e-01, 5.1705e-02,
    0.0000e+00, 0.0000e+00, 3.0565e-01, 4.5431e-01, 0.0000e+00, 4.4742e-02,
    2.3583e-01, 8.6285e-01, 0.0000e+00, 6.5763e-02, 0.0000e+00, 7.2151e-01,
    7.7982e-01, 5.3851e-01, 0.0000e+00, 0.0000e+00, 3.1895e-01, 0.0000e+00,
    0.0000e+00, 8.2074e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 5.9684e-01, 0.0000e+00, 0.0000e+00, 2.1435e-03,
    0.0000e+00, 4.0795e-02, 6.0204e-01, 5.3779e-02, 3.4706e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.0793e-01,
    0.0000e+00, 2.8389e-01, 9.6737e-02, 0.0000e+00, 4.0340e-01, 5.9990e-01,
    9.6217e-01, 1.7721e-01, 0.0000e+00, 0.0000e+00, 2.1650e-01, 0.0000e+00,
    3.6605e-01, 1.4687e-01, 0.0000e+00, 0.0000e+00, 3.9823e-01, 3.9444e-01,
    0.0000e+00, 0.0000e+00, 2.0717e-01, 3.5793e-01, 0.0000e+00, 0.0000e+00,
    5.3949e-01, 3.0678e-01, 0.0000e+00, 5.6074e-01, 0.0000e+00, 5.5505e-01,
    0.0000e+00, 6.2500e-01, 1.4662e-01, 1.7523e-01, 1.1858e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 4.8805e-01, 0.0000e+00, 0.0000e+00, 2.7573e-01,
    7.1308e-02, 0.0000e+00, 5.8064e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 7.5257e-02, 1.0648e-01, 0.0000e+00,
    4.6254e-02, 0.0000e+00, 2.1399e-02, 4.7951e-01, 0.0000e+00, 0.0000e+00,
    2.7274e-01, 1.7287e-01, 0.0000e+00, 4.2774e-01, 0.0000e+00, 0.0000e+00,
    2.3287e-01, 0.0000e+00, 0.0000e+00, 1.0729e-01, 0.0000e+00, 6.1395e-01,
    8.5825e-01, 0.0000e+00, 1.5815e-01, 0.0000e+00, 7.7004e-02, 0.0000e+00,
    4.0929e-01, 3.3197e-01, 0.0000e+00, 0.0000e+00, 3.9544e-01, 6.5016e-02,
    0.0000e+00, 3.9518e-01, 0.0000e+00, 0.0000e+00, 3.6473e-01, 6.8897e-01,
    2.2457e-01, 4.7769e-01, 1.2626e-01, 0.0000e+00, 0.0000e+00, 3.9260e-01,
    0.0000e+00, 0.0000e+00, 2.5657e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 2.0502e-01, 3.7539e-01, 8.7846e-01, 6.4727e-02,
    1.1631e-01, 1.1159e-01, 9.1746e-01, 1.7563e-01, 7.0101e-02, 1.5769e-01,
    0.0000e+00, 0.0000e+00, 4.7747e-01, 0.0000e+00, 0.0000e+00, 2.3046e-02,
    6.8920e-02, 0.0000e+00, 8.2851e-01, 1.2707e-01, 0.0000e+00, 0.0000e+00,
    2.7831e-01, 0.0000e+00, 7.0561e-01, 0.0000e+00, 0.0000e+00, 5.4838e-01,
    0.0000e+00, 0.0000e+00, 6.3581e-02, 1.5840e-02, 2.3359e-01, 0.0000e+00,
    1.1576e-01, 1.8558e-02, 3.4576e-01, 5.1232e-02, 0.0000e+00, 3.6624e-01,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.3642e-01,
    0.0000e+00, 0.0000e+00, 2.1666e-01, 6.6057e-01, 2.2358e-01, 2.1319e-01,
    0.0000e+00, 0.0000e+00, 1.5222e-01, 0.0000e+00, 4.0792e-01, 5.0303e-01,
    3.2759e-01, 5.0508e-01, 6.1877e-01, 0.0000e+00, 5.3355e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 2.7622e-01, 0.0000e+00, 5.2218e-01,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 7.0829e-01, 5.4879e-02, 0.0000e+00,
    0.0000e+00, 2.7878e-01, 1.4440e-01, 3.2567e-02, 5.8117e-01, 0.0000e+00,
    2.1245e-02, 0.0000e+00, 7.7204e-01, 6.7993e-01, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 5.7836e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    1.1558e-01, 7.0709e-03, 4.5603e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 9.1948e-01, 0.0000e+00, 6.2814e-01, 0.0000e+00, 0.0000e+00,
    3.3349e-01, 3.0334e-01, 4.3112e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 2.0491e-02, 2.3031e-03, 0.0000e+00, 0.0000e+00, 4.4998e-01,
    3.3105e-01, 0.0000e+00, 6.7470e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    1.7714e-01, 0.0000e+00, 6.5216e-01, 2.7070e-01, 3.8228e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 2.1267e-01, 0.0000e+00, 4.5109e-01,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.5450e-01, 4.0487e-02,
    0.0000e+00, 1.2057e+00, 1.6021e-01, 0.0000e+00, 0.0000e+00, 1.4730e-01,
    3.1517e-01, 3.1166e-01, 0.0000e+00, 5.8983e-02, 0.0000e+00, 3.9183e-02,
    0.0000e+00, 5.9395e-01, 1.8784e-01, 7.6888e-01, 0.0000e+00, 0.0000e+00,
    8.3103e-02, 2.5260e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.5295e-01,
    6.6527e-01, 0.0000e+00, 4.2463e-01, 4.4498e-01, 3.5978e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 2.0312e-01, 0.0000e+00, 1.9440e-01, 0.0000e+00,
    7.0238e-02, 0.0000e+00, 8.2609e-01, 0.0000e+00, 1.5140e-01, 1.8528e-01,
    0.0000e+00, 1.0557e-01, 4.4546e-01, 0.0000e+00, 2.4473e-01, 2.0740e-01,
    0.0000e+00, 0.0000e+00, 2.7884e-01, 7.4443e-03, 0.0000e+00, 1.9991e-01,
    0.0000e+00, 0.0000e+00, 2.6190e-01, 0.0000e+00, 2.4432e-01, 3.0944e-01,
    1.7316e-01, 4.4882e-01, 0.0000e+00, 2.7011e-01, 0.0000e+00, 8.2411e-02,
    2.6806e-01, 4.2293e-01, 5.7905e-01, 3.9226e-02, 0.0000e+00, 0.0000e+00,
    7.4738e-01, 0.0000e+00, 6.4801e-01, 0.0000e+00, 1.1211e-01, 1.9855e-01,
    2.1242e-01, 0.0000e+00, 0.0000e+00, 5.3369e-01, 2.1392e-01, 0.0000e+00,
    0.0000e+00, 5.6768e-02, 7.3865e-01, 0.0000e+00, 4.4628e-01, 5.4678e-01,
    0.0000e+00, 1.2406e+00, 5.4774e-01, 0.0000e+00, 8.3491e-01, 0.0000e+00,
    2.8482e-03, 0.0000e+00, 2.3726e-02, 4.4714e-01, 0.0000e+00, 4.3510e-01,
    8.8795e-03, 1.1618e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.3650e-01,
    0.0000e+00, 2.5490e-01, 0.0000e+00, 1.0139e+00, 2.2726e-01, 0.0000e+00,
    0.0000e+00, 3.7998e-01, 0.0000e+00, 5.2665e-01, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.8971e-01, 3.7777e-02,
    2.4558e-02, 0.0000e+00, 6.1270e-01, 5.3864e-01, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 4.3362e-01, 0.0000e+00, 3.1963e-01, 0.0000e+00, 8.6494e-02,
    8.9975e-02, 0.0000e+00, 3.6914e-01, 0.0000e+00, 0.0000e+00, 2.0640e-02,
    0.0000e+00, 4.0372e-01, 7.4526e-02, 0.0000e+00, 0.0000e+00, 1.4382e-01,
    0.0000e+00, 1.8750e-01, 1.7591e-01, 0.0000e+00, 1.8670e-02, 0.0000e+00,
    2.7231e-01, 4.2284e-01, 1.0397e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 1.1628e-01, 0.0000e+00, 0.0000e+00, 2.6968e-01, 0.0000e+00,
    0.0000e+00, 3.1947e-01, 0.0000e+00, 1.6322e-01, 3.1021e-02, 0.0000e+00,
    4.2218e-01, 0.0000e+00, 0.0000e+00, 5.2111e-01, 0.0000e+00, 0.0000e+00,
    2.7029e-01, 1.4907e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.9614e-01,
    0.0000e+00, 0.0000e+00, 6.8842e-02, 0.0000e+00, 3.9265e-01, 1.5416e-02,
    0.0000e+00, 2.6446e-01, 0.0000e+00, 2.4775e-01, 0.0000e+00, 2.1911e-02,
    0.0000e+00, 4.4974e-01, 3.8515e-02, 2.5485e-02, 1.6000e-01, 0.0000e+00,
    0.0000e+00, 7.2951e-01, 0.0000e+00, 2.2118e-01, 6.5213e-01, 0.0000e+00,
    2.9677e-01, 0.0000e+00, 6.0142e-01, 2.0937e-01, 4.9422e-01, 1.2792e-01,
    0.0000e+00, 1.1312e-02, 0.0000e+00, 0.0000e+00, 2.1873e-01, 0.0000e+00,
    2.7908e-02, 0.0000e+00, 4.2024e-02, 8.2473e-01, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 7.6014e-02, 2.0516e-01, 3.6212e-01, 6.7401e-02, 8.0160e-01,
    1.5580e-01, 4.6914e-01, 3.8585e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 4.6813e-01, 4.7961e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 3.6201e-02, 2.9699e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 2.2658e-01, 0.0000e+00, 3.2805e-02, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 2.1642e-01, 8.5423e-03, 0.0000e+00, 8.3172e-02,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.6319e-01,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.4710e-01, 1.1487e-01,
    0.0000e+00, 2.1976e-01, 0.0000e+00, 0.0000e+00, 3.4147e-03, 0.0000e+00,
    0.0000e+00, 7.4752e-01, 1.1369e-01, 2.8925e-01, 0.0000e+00, 0.0000e+00,
    6.9137e-03, 0.0000e+00, 6.8495e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 8.7484e-02, 3.6542e-02, 7.0685e-01, 0.0000e+00, 5.1035e-01,
    0.0000e+00, 4.5207e-01, 0.0000e+00, 9.4534e-02, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 1.6015e-01, 3.9820e-02, 1.9649e-01, 5.7938e-01, 0.0000e+00,
    5.0997e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 1.6138e-01, 9.7867e-02, 5.1788e-03, 4.0551e-02, 3.3331e-01,
    2.0930e-01, 1.7489e-01, 0.0000e+00, 5.2685e-01, 1.5958e-01, 0.0000e+00,
    1.8879e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.0311e-01,
    3.4122e-01, 0.0000e+00, 0.0000e+00, 5.3115e-01, 1.6489e-01, 0.0000e+00,
    1.2003e-01, 1.4094e-01, 5.1902e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    2.7128e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8408e-01, 1.0612e-01,
    2.6369e-01, 5.6802e-01, 4.3276e-01, 0.0000e+00, 0.0000e+00, 1.7773e-02,
    0.0000e+00, 0.0000e+00, 8.4720e-01, 0.0000e+00, 0.0000e+00, 2.0917e-01,
    0.0000e+00, 5.3588e-01, 1.4548e-01, 2.3757e-01, 0.0000e+00, 1.7626e-01,
    0.0000e+00, 0.0000e+00, 9.9600e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    7.1133e-01, 0.0000e+00, 0.0000e+00, 3.2668e-01, 4.6632e-01, 2.9542e-01,
    5.7628e-01, 0.0000e+00, 0.0000e+00, 1.6011e-01, 2.2801e-01, 2.4029e-01,
    4.0798e-02, 0.0000e+00, 2.4332e-01, 4.1634e-01, 5.6399e-01, 0.0000e+00,
    0.0000e+00, 2.7964e-01, 0.0000e+00, 0.0000e+00, 4.5204e-02, 2.3471e-01,
    5.9523e-01, 5.9002e-01, 3.8267e-01, 0.0000e+00, 6.9757e-01, 0.0000e+00,
    4.1682e-01, 1.8667e-01, 0.0000e+00, 1.6129e-01, 0.0000e+00, 2.1806e-01,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 4.3339e-01, 4.6584e-01, 0.0000e+00,
    2.3673e-01, 2.9363e-02, 1.3844e-01, 0.0000e+00, 6.9891e-01, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 8.0242e-02, 5.0152e-01, 0.0000e+00, 0.0000e+00,
    2.4636e-01, 2.1743e-01, 0.0000e+00, 0.0000e+00),
    T(3.4477e-01, 5.3790e-01, 0.0000e+00, 0.0000e+00, 2.2307e-01, 0.0000e+00,
      9.5184e-02, 0.0000e+00, 4.9757e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      1.0260e-01, 3.7487e-01, 4.1819e-01, 6.2474e-01, 1.4737e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 3.6250e-01, 0.0000e+00, 0.0000e+00, 1.5794e-01,
      6.9416e-01, 8.9911e-01, 3.7086e-01, 0.0000e+00, 4.5447e-01, 1.0841e-01,
      2.4890e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.7963e-01,
      0.0000e+00, 5.9176e-01, 6.4291e-01, 3.0385e-01, 0.0000e+00, 3.8872e-01,
      0.0000e+00, 0.0000e+00, 3.6584e-01, 1.8300e-01, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 5.3081e-01, 0.0000e+00, 6.3115e-01, 0.0000e+00, 0.0000e+00,
      9.0696e-02, 0.0000e+00, 2.2288e-02, 3.9550e-01, 1.7167e-01, 0.0000e+00,
      2.3810e-01, 0.0000e+00, 2.1348e-01, 1.6472e-01, 0.0000e+00, 9.7072e-01,
      0.0000e+00, 6.7466e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.5414e-02,
      0.0000e+00, 5.0091e-01, 4.1359e-01, 0.0000e+00, 0.0000e+00, 2.0857e-01,
      0.0000e+00, 1.9868e-01, 7.8524e-02, 2.6530e-01, 3.0148e-01, 3.9244e-02,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      4.6995e-01, 0.0000e+00, 4.5016e-01, 4.2471e-01, 0.0000e+00, 8.5689e-01,
      2.9450e-01, 1.0400e-01, 5.3195e-01, 0.0000e+00, 6.8811e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 4.7606e-02, 0.0000e+00, 5.7919e-02, 6.7971e-01,
      0.0000e+00, 1.9129e-01, 7.7240e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      1.9061e-01, 2.6352e-01, 0.0000e+00, 0.0000e+00, 7.0253e-01, 0.0000e+00,
      5.6434e-01, 0.0000e+00, 5.5095e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      5.1218e-01, 0.0000e+00, 0.0000e+00, 2.7837e-01, 5.1009e-03, 1.2288e-01,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4556e-01, 0.0000e+00, 0.0000e+00,
      1.0213e+00, 4.3464e-02, 5.5964e-01, 0.0000e+00, 4.2587e-01, 0.0000e+00,
      1.8511e-01, 0.0000e+00, 5.8170e-01, 2.4356e-01, 1.7729e-02, 0.0000e+00,
      0.0000e+00, 6.4470e-01, 0.0000e+00, 7.3560e-02, 7.5141e-02, 0.0000e+00,
      1.8372e-01, 3.0369e-02, 4.1808e-01, 0.0000e+00, 3.8276e-01, 0.0000e+00,
      5.5623e-02, 0.0000e+00, 2.3709e-01, 9.9455e-02, 1.7909e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.2145e-02,
      0.0000e+00, 5.5455e-01, 4.6552e-01, 1.2873e-01, 1.6117e-01, 0.0000e+00,
      8.1001e-01, 3.3079e-01, 3.7088e-01, 4.2121e-01, 3.6645e-01, 0.0000e+00,
      0.0000e+00, 1.4828e-02, 0.0000e+00, 1.1300e-01, 0.0000e+00, 0.0000e+00,
      3.6897e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.0542e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 3.5009e-01, 5.1069e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 1.3777e-01, 0.0000e+00, 3.4843e-01, 4.7614e-01,
      8.1894e-01, 6.0401e-01, 1.1808e+00, 3.8150e-01, 0.0000e+00, 3.8185e-01,
      1.3258e-01, 4.7142e-01, 0.0000e+00, 0.0000e+00, 1.6631e-01, 0.0000e+00,
      0.0000e+00, 2.8422e-01, 2.3445e-01, 0.0000e+00, 4.0709e-01, 0.0000e+00,
      3.3331e-02, 0.0000e+00, 7.4624e-01, 0.0000e+00, 0.0000e+00, 1.4074e-01,
      2.6273e-02, 1.7291e-01, 0.0000e+00, 3.1316e-01, 7.9369e-01, 5.0961e-01,
      0.0000e+00, 6.3910e-01, 4.4863e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      3.1543e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.0503e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 1.4077e-02, 2.3304e-01, 4.5136e-01, 3.1046e-01,
      0.0000e+00, 0.0000e+00, 4.6712e-02, 3.0973e-02, 3.9185e-02, 0.0000e+00,
      7.3608e-02, 0.0000e+00, 0.0000e+00, 2.7621e-01, 0.0000e+00, 5.7861e-01,
      5.9262e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 1.9767e-01, 2.8329e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 1.6375e-01, 5.8476e-01, 3.0416e-01, 7.8206e-02,
      1.3225e-02, 5.0259e-04, 3.7120e-01, 0.0000e+00, 2.6165e-01, 5.1930e-02,
      0.0000e+00, 0.0000e+00, 3.0789e-01, 5.3606e-01, 0.0000e+00, 1.5168e-01,
      1.7575e-01, 9.5358e-01, 0.0000e+00, 2.5855e-01, 0.0000e+00, 6.6173e-01,
      6.5663e-01, 4.3380e-01, 0.0000e+00, 0.0000e+00, 2.6316e-01, 0.0000e+00,
      0.0000e+00, 8.7554e-01, 2.6346e-02, 0.0000e+00, 4.4752e-02, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 3.9447e-01, 0.0000e+00, 0.0000e+00, 3.7064e-03,
      0.0000e+00, 2.2273e-01, 5.2556e-01, 1.7410e-01, 2.8065e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4782e-01,
      0.0000e+00, 3.2743e-01, 2.0456e-01, 0.0000e+00, 4.7197e-01, 4.0365e-01,
      8.1541e-01, 4.3598e-02, 0.0000e+00, 0.0000e+00, 1.9783e-01, 0.0000e+00,
      2.8507e-01, 7.6302e-02, 0.0000e+00, 0.0000e+00, 3.6714e-01, 2.0126e-01,
      0.0000e+00, 0.0000e+00, 8.3921e-02, 8.6749e-02, 1.1394e-01, 0.0000e+00,
      5.0907e-01, 4.0542e-01, 2.0926e-01, 2.9741e-01, 0.0000e+00, 6.4717e-01,
      0.0000e+00, 5.4612e-01, 0.0000e+00, 1.5081e-01, 9.7054e-02, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 4.8004e-01, 0.0000e+00, 0.0000e+00, 3.0902e-01,
      1.7005e-01, 0.0000e+00, 4.2053e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 2.3425e-02, 0.0000e+00, 2.9508e-02,
      1.2342e-01, 6.6151e-02, 1.1795e-01, 4.3885e-01, 0.0000e+00, 6.1384e-02,
      3.2079e-01, 3.3351e-01, 0.0000e+00, 4.7612e-01, 0.0000e+00, 0.0000e+00,
      5.2961e-02, 0.0000e+00, 0.0000e+00, 1.6878e-01, 0.0000e+00, 5.6907e-01,
      9.8154e-01, 0.0000e+00, 2.4978e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      4.6588e-01, 3.4975e-01, 0.0000e+00, 0.0000e+00, 3.4350e-01, 0.0000e+00,
      0.0000e+00, 4.6224e-01, 1.1569e-01, 0.0000e+00, 2.8777e-01, 5.3360e-01,
      2.5801e-01, 3.6288e-01, 3.0857e-01, 0.0000e+00, 0.0000e+00, 4.2739e-01,
      0.0000e+00, 0.0000e+00, 4.2144e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 1.3603e-01, 1.5070e-01, 9.2289e-01, 5.9535e-02,
      8.4853e-02, 1.3326e-01, 9.8809e-01, 1.7581e-01, 3.1391e-01, 3.1574e-01,
      0.0000e+00, 0.0000e+00, 2.1502e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      4.9106e-02, 0.0000e+00, 7.7161e-01, 1.6317e-01, 8.3655e-02, 0.0000e+00,
      4.1273e-01, 0.0000e+00, 7.3198e-01, 0.0000e+00, 0.0000e+00, 5.1957e-01,
      0.0000e+00, 0.0000e+00, 2.0662e-01, 2.0251e-01, 1.9196e-01, 0.0000e+00,
      2.1973e-01, 1.0293e-01, 5.1585e-01, 5.0458e-02, 0.0000e+00, 2.6205e-01,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 1.2768e-01, 4.1336e-01, 4.1304e-02, 1.3741e-01,
      0.0000e+00, 6.8382e-02, 1.9069e-01, 0.0000e+00, 5.1551e-01, 4.3364e-01,
      5.5200e-01, 3.3187e-01, 6.0491e-01, 0.0000e+00, 4.2924e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4376e-01, 0.0000e+00, 4.4113e-01,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 8.1884e-01, 8.9262e-02, 0.0000e+00,
      0.0000e+00, 4.9353e-02, 4.0501e-01, 3.6185e-02, 5.6896e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 6.9891e-01, 6.5879e-01, 0.0000e+00, 1.1593e-02,
      0.0000e+00, 0.0000e+00, 5.9424e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      2.5864e-01, 6.6448e-02, 5.2231e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 7.5423e-01, 0.0000e+00, 6.6645e-01, 1.2906e-01, 0.0000e+00,
      3.3155e-01, 2.8875e-01, 4.6569e-01, 2.2770e-01, 0.0000e+00, 2.3122e-01,
      0.0000e+00, 2.2629e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.4923e-01,
      3.6517e-01, 0.0000e+00, 9.5871e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      4.2188e-01, 1.5313e-03, 5.2581e-01, 1.9032e-01, 1.1789e-01, 0.0000e+00,
      0.0000e+00, 8.2999e-02, 0.0000e+00, 1.4839e-01, 0.0000e+00, 4.3182e-01,
      0.0000e+00, 1.2897e-01, 4.4780e-02, 0.0000e+00, 9.3899e-02, 0.0000e+00,
      0.0000e+00, 1.1690e+00, 3.1387e-01, 0.0000e+00, 0.0000e+00, 2.5334e-01,
      2.6436e-01, 1.0286e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 5.2249e-01, 2.0861e-01, 7.4454e-01, 0.0000e+00, 0.0000e+00,
      2.5486e-01, 1.8852e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.9914e-01,
      9.0941e-01, 0.0000e+00, 2.2463e-01, 3.8639e-01, 2.5263e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 1.8697e-01, 0.0000e+00, 1.5987e-01, 0.0000e+00,
      8.4178e-02, 0.0000e+00, 8.1653e-01, 1.3396e-02, 0.0000e+00, 6.6077e-02,
      0.0000e+00, 0.0000e+00, 4.0837e-01, 0.0000e+00, 2.1575e-01, 2.2600e-01,
      0.0000e+00, 0.0000e+00, 4.6028e-01, 1.7596e-01, 0.0000e+00, 6.6201e-03,
      0.0000e+00, 0.0000e+00, 2.5426e-01, 0.0000e+00, 1.4503e-01, 2.3080e-01,
      1.4868e-01, 3.3499e-01, 6.6406e-02, 2.2613e-01, 0.0000e+00, 3.2301e-01,
      8.1848e-03, 3.4828e-01, 4.8847e-01, 1.2102e-01, 0.0000e+00, 0.0000e+00,
      6.7854e-01, 0.0000e+00, 5.0661e-01, 0.0000e+00, 2.6678e-01, 0.0000e+00,
      2.8975e-01, 6.6258e-02, 0.0000e+00, 4.7068e-01, 1.8406e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 6.5444e-01, 1.5379e-01, 4.1505e-01, 5.8430e-01,
      0.0000e+00, 1.2742e+00, 3.1801e-01, 0.0000e+00, 7.7948e-01, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 5.2493e-01, 0.0000e+00, 4.6612e-01,
      0.0000e+00, 1.7579e-01, 5.7805e-02, 0.0000e+00, 0.0000e+00, 4.7099e-01,
      0.0000e+00, 4.4843e-01, 7.6184e-02, 9.4211e-01, 1.2329e-01, 0.0000e+00,
      0.0000e+00, 2.4681e-01, 0.0000e+00, 5.4552e-01, 0.0000e+00, 0.0000e+00,
      3.9235e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0743e-01, 8.1704e-02,
      0.0000e+00, 0.0000e+00, 5.3216e-01, 4.1451e-01, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 2.9729e-01, 0.0000e+00, 4.1537e-01, 0.0000e+00, 8.4075e-02,
      0.0000e+00, 0.0000e+00, 2.1437e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 1.7877e-01, 6.1475e-02, 0.0000e+00, 0.0000e+00, 2.3879e-01,
      0.0000e+00, 2.3353e-01, 0.0000e+00, 0.0000e+00, 8.7434e-02, 0.0000e+00,
      2.5284e-01, 4.5843e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.6319e-01, 0.0000e+00,
      0.0000e+00, 1.7851e-01, 0.0000e+00, 5.4940e-02, 0.0000e+00, 0.0000e+00,
      7.6305e-01, 0.0000e+00, 0.0000e+00, 5.3016e-01, 0.0000e+00, 6.4501e-02,
      3.2596e-01, 2.2609e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.6258e-01,
      0.0000e+00, 0.0000e+00, 3.6065e-02, 0.0000e+00, 4.3639e-01, 0.0000e+00,
      0.0000e+00, 2.8636e-01, 0.0000e+00, 1.2189e-01, 0.0000e+00, 1.9672e-02,
      0.0000e+00, 2.0239e-01, 6.8883e-02, 0.0000e+00, 2.0419e-01, 1.0278e-01,
      0.0000e+00, 6.3682e-01, 0.0000e+00, 0.0000e+00, 3.5179e-01, 0.0000e+00,
      3.3570e-01, 0.0000e+00, 6.4269e-01, 1.8769e-01, 7.2755e-01, 2.6766e-01,
      0.0000e+00, 5.2196e-02, 0.0000e+00, 0.0000e+00, 1.8828e-01, 0.0000e+00,
      1.9833e-01, 0.0000e+00, 1.4594e-01, 6.5152e-01, 0.0000e+00, 0.0000e+00,
      1.3182e-01, 0.0000e+00, 0.0000e+00, 5.1446e-01, 0.0000e+00, 8.8741e-01,
      3.6072e-02, 2.8367e-01, 2.0990e-01, 0.0000e+00, 0.0000e+00, 2.9495e-01,
      9.2837e-02, 2.7265e-01, 4.5158e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 5.1664e-02, 1.8977e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 6.8549e-02, 0.0000e+00, 1.4437e-01, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 3.2509e-02, 2.7062e-02, 0.0000e+00, 1.5187e-01,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 8.5417e-01,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.3508e-01, 3.4785e-01,
      0.0000e+00, 9.3501e-02, 0.0000e+00, 0.0000e+00, 1.0156e-01, 0.0000e+00,
      0.0000e+00, 7.1037e-01, 1.9067e-01, 1.1440e-01, 0.0000e+00, 0.0000e+00,
      1.4178e-01, 1.5496e-01, 6.0515e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 9.0447e-02, 6.1297e-01, 0.0000e+00, 6.6387e-01,
      0.0000e+00, 1.3241e-01, 0.0000e+00, 1.0057e-01, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 4.5816e-01, 9.3208e-02, 1.3828e-01, 3.8518e-01, 0.0000e+00,
      6.0722e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 3.0409e-01, 6.1371e-02, 0.0000e+00, 9.8232e-02, 2.3291e-01,
      3.3932e-01, 5.2183e-02, 0.0000e+00, 5.2186e-01, 8.3608e-02, 0.0000e+00,
      2.1225e-01, 0.0000e+00, 0.0000e+00, 8.4130e-02, 0.0000e+00, 1.3490e-01,
      2.8295e-01, 0.0000e+00, 0.0000e+00, 4.7005e-01, 1.1182e-01, 0.0000e+00,
      0.0000e+00, 1.2451e-01, 3.3872e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
      2.5682e-01, 7.9506e-02, 0.0000e+00, 6.3264e-02, 0.0000e+00, 0.0000e+00,
      4.0924e-01, 5.9037e-01, 5.5715e-01, 0.0000e+00, 0.0000e+00, 1.0825e-01,
      0.0000e+00, 0.0000e+00, 7.5731e-01, 0.0000e+00, 0.0000e+00, 3.0620e-01,
      0.0000e+00, 5.8681e-01, 0.0000e+00, 1.4702e-01, 0.0000e+00, 0.0000e+00,
      0.0000e+00, 0.0000e+00, 1.5751e-02, 0.0000e+00, 0.0000e+00, 1.4469e-02,
      5.7545e-01, 0.0000e+00, 0.0000e+00, 2.4175e-01, 5.6689e-01, 4.0856e-01,
      5.2724e-01, 2.6466e-02, 0.0000e+00, 0.0000e+00, 3.6880e-01, 6.0492e-02,
      0.0000e+00, 0.0000e+00, 1.7004e-01, 4.8291e-01, 6.7784e-01, 1.4573e-02,
      0.0000e+00, 7.9193e-02, 0.0000e+00, 0.0000e+00, 9.3218e-02, 4.6077e-01,
      4.5760e-01, 4.3444e-01, 1.9278e-01, 0.0000e+00, 7.7304e-01, 0.0000e+00,
      5.4956e-01, 5.2741e-01, 0.0000e+00, 1.9542e-01, 0.0000e+00, 5.5032e-02,
      0.0000e+00, 0.0000e+00, 0.0000e+00, 3.7545e-01, 6.4552e-01, 0.0000e+00,
      2.8725e-01, 1.2046e-01, 6.8067e-02, 0.0000e+00, 5.6154e-01, 0.0000e+00,
      5.8774e-02, 0.0000e+00, 3.0693e-02, 3.3855e-01, 0.0000e+00, 0.0000e+00,
      9.2428e-02, 4.1654e-01, 0.0000e+00, 0.0000e+00)))

    val layer = new FPNPredictor(num_class, in_channels)

    val out = layer.forward(input)

    println("output")
  }
}
