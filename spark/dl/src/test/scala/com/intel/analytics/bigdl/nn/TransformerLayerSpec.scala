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

class TransformerLayerSpec extends FlatSpec with Matchers {
  "tranformer decode stack" should "work correctly" in {
    val vocabSize = 10
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 1
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](vocabSize,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)

    val input1 = Input[Float]()
    val input2 = Input[Float]()

    val blockOutput = transformer.decodeStack(num_hidden_layers, input1, input2)
    val block = Graph(Array(input1, input2), blockOutput)
    val paramsTable = block.getParametersTable()

    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.12254566, -0.3492695, 0.6760147, 0.4690166),
            T(-0.70616156, -0.7172935, -0.70902413, -0.7268282),
            T(-0.17867321, 0.03752673, 0.21406537, -0.84105927),
            T(-0.40054652, 0.01422167, 0.49654406, -0.62966037))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.80201703, 0.29880065, 0.8191585, 0.393151),
            T(-0.43785518, 0.02502167, -0.85530514, 0.86387163),
            T( 0.07737422, 0.34640843, 0.5547114, 0.12658376),
            T( 0.6287202, -0.7140273, -0.08061278, -0.3983137))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.14568096, 0.8488055, -0.38585222, -0.42583144),
            T(-0.35776895, 0.00440949, 0.76952034, 0.7039148),
            T(-0.4635923, -0.5273898, 0.36311775, 0.21081167),
            T(-0.04171634, 0.24859089, 0.03242427, -0.01675642))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T( 0.8254406, 0.7399195, -0.76593506, -0.38950253),
              T( 0.51289314, 0.1285783, -0.24543494, -0.7138509),
              T(-0.34158242, -0.37842813, -0.5111934, 0.5966528),
              T( 0.39076942, -0.7022542, 0.8254971, -0.50844))).t())
      } else if (i.toString contains "_filter_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T( 0.4929167, -0.5465611, 0.4262464),
              T( 0.5161569, -0.6786176, 0.37465477),
              T( 0.35582626, 0.43647707, -0.23218763),
              T( 0.7624726, 0.28653884, 0.20991063))).transpose(1, 2))
      } else if (i.toString contains "_output_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.9037433, 0.6076299, 0.6593666, -0.06372046),
              T( 0.58014977, 0.6601094, -0.72481453, 0.89943814),
              T( 0.02975523, -0.4040287, 0.6437061, -0.2594086))).transpose(1, 2))
      }
    }

    val input = Tensor[Float](Tensor[Float](
      T(T(T( 2.43651805, -0.91763462, -0.79225763, -1.60945293),
        T( 1.29811144, -3.45230805, 2.61721765, -1.14181035),
        T( 0.47855864, -0.37405556, 2.19316191, -3.09021106),
        T(-0.48362581, -0.57608153, 1.70065416, -1.6498369),
        T(-0.25864231, -1.31678763, 0.06332062, 0.87422282),
        T(-1.65092877, 1.71708556, 1.35238608, 0.75374151)),
        T(T( 1.35128392, -1.02559179, -0.18433534, -1.40365415),
          T(-0.40183212, 0.7955332, -1.03749113, -0.59513029),
          T(-1.03075905, -1.26780846, -1.0068692, -0.0189969),
          T(-1.67596552, 0.35162355, 2.48970327, 1.11306624),
          T(-0.28775333, -1.33144345, -1.12073744, 2.5386819),
          T( 0.07621163, -0.95549347, 0.28637323, 3.1503827)))))
    val bias = Tensor[Float](Tensor[Float](
      T(T(T(T( 0.12015895, 0.61720311, 0.30017032, -0.35224985, -1.1425182, -0.34934272),
        T(-0.20889423, 0.58662319, 0.83898341, 0.93110208, 0.28558733, 0.88514116),
        T(-0.75439794, 1.25286816, 0.51292982, -0.29809284, 0.48851815, -0.07557171),
        T( 1.13162939, 1.51981682, 2.18557541, -1.39649634, -1.44411381, -0.50446586),
        T( 0.16003707, 0.87616892, 0.31563495, -2.02220122, -0.30620401, 0.82797464),
        T( 0.23009474, 0.76201118, -0.22232814, -0.20075807, 0.18656139, 0.41005165))))
    ))

    val expectedOutput = Tensor[Float](
      T(T(T( 1.6739436, -0.5742816, -0.18686886, -0.91279316),
        T( 0.56332755, -1.6895478, 0.8744801, 0.25174013),
        T( 0.18294929, -0.03678755, 1.333065, -1.4792268),
        T(-0.83871794, 0.09105678, 1.6003608, -0.8526995),
        T(-0.6227458, 0.06268612, -1.0336334, 1.593693),
        T(-1.6069404, 0.70157117, -0.05510008, 0.9604694)),
        T(T( 1.500092, -0.12251449, -0.06195105, -1.3156266),
          T( 0.88058877, 0.88686943, -0.2218959, -1.5455623),
          T(-1.73186, 0.59709984, 0.5559552, 0.5788053),
          T(-1.7018749, 0.8331325, 0.30757982, 0.56116235),
          T(-0.5026365, -0.1983719, -0.96522677, 1.6662351),
          T(-0.56770575, -0.17644365, -0.92594254, 1.6700919)))
    )

    val output = block.forward(T(input, bias))
    output should be(expectedOutput)

    val gradInput = block.backward(T(input, bias), output)

    println("done")
  }

  "tranformer for translation" should "work correctly" in {
    val vocabSize = 16
    val hiddenSize = 4
    val filterSize = 8
    val numHeads = 1
    val num_hidden_layers = 1
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val transformer = new TransformerLayer[Float](vocabSize,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout, problem = Translation)

    val attention0 = transformer.model("encode_self_attention_0/self_attention").get
    val ffn0 = transformer.model("encode_ffn_0/ffn").get

    val attention1 = transformer.model("decode_self_attention_0/self_attention").get
    val ffn1 = transformer.model("decode_ffn_0/ffn").get
    val attention2 = transformer.model("decode_encdec_attention_0/encdec_attention").get

    var paramsTable = attention0.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.6719899, 0.29741684, -0.6073703, 0.58373296),
            T(0.28487056, 0.12325107, -0.18469666, -0.3146433),
            T(0.60392314, 0.65988046, 0.50996345, -0.19420744),
            T(0.40057203, -0.9149872, 0.10390836, 0.97260743))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.33549386, 0.88536686, -0.30634838, 0.05587747),
            T(0.61110026, -0.66457653, -0.34049615, -0.14537863),
            T(0.653832, 0.74835855, 0.76725274, -0.6947307),
            T(0.49148628, -0.07944908, -0.845008, 0.6068878))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.24006118, 0.6792713, 0.22704636, 0.49668023),
            T(0.53909445, -0.32836607, 0.25972122, 0.5554116),
            T(-0.4319324, 0.43911168, 0.20273127, -0.24734582),
            T(0.23329619, -0.3165343, 0.40053207, -0.34865358))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.5211139, -0.3813012, 0.34638476, -0.21196833),
              T(0.1121366, -0.3850857, 0.15838127, -0.46018872),
              T(0.42922392, 0.49836066, -0.00889128, -0.20409666),
              T(-0.0800805, 0.6680052, 0.11346864, -0.3564058))).t())
      }
    }

    paramsTable = ffn0.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_filter_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.42055795, -0.345141, -0.77015144, 1.0128733, -0.2070824,
              0.41457736, -0.27325338, 0.37545303),
              T(0.83861953, 0.49639514, 0.10912374, 0.4054078, 0.01117581,
                0.4838021, 0.47710165, 0.23820893),
              T(-0.37739983, -0.3799013, 0.26106557, -0.02527841, -0.09814293,
                0.15995328, 0.76590466, -0.38680843),
              T(0.22057502, 0.4438025, 0.18568423, 0.2206358, -0.5293094,
                -0.07671213, -0.5392774, -0.26026365))).transpose(1, 2))
      } else if (i.toString contains "_output_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.15800391, 0.00911217, 0.5716306, -0.4307602),
              T(-0.17119521, 0.45397595, -0.15994692, 0.1173245),
              T(0.02792565, 0.1785465, 0.03194377, -0.2635249),
              T(-0.5619625, 0.34994912, 0.2134058, 0.17008546),
              T(-0.16928878, -0.04155388, -0.00634552, 0.10220164),
              T(-0.19378763, 0.60514146, 0.31211597, 0.32819757),
              T(-0.12504072, -0.5004057, -0.53571004, -0.6392757),
              T(-0.06203287, 0.25287995, 0.32892716, 0.11961207))).transpose(1, 2))
      }
    }

    paramsTable = attention1.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.58024985, -0.48674917, -0.1278461, -0.1681186),
            T(1.0511181, 0.50676775, -0.49831128, -0.13611957),
            T(0.4512829, 0.00988893, 0.35473365, -0.4541598),
            T(-0.01564673, -0.06611676, 0.20534483, -0.13249157))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(0.25792515, 0.8091696, -1.1157143, -0.48759258),
            T(0.2797681, -0.61634296, 0.29310933, 0.3868902),
            T(-0.22521666, -0.08918925, 0.17066494, 0.06447314),
            T(-0.14935619, -0.05546288, -1.134581, 0.33467665))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.05646669, 0.2533887, 0.9146523, 0.09979013),
            T(-0.03409033, 0.9656157, -0.00790233, 0.22394712),
            T(0.44499645, -0.41030893, -0.40253338, -0.541713),
            T(0.63082635, 0.05910337, 0.26689664, 0.06098993))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(0.07528905, -0.6294302, -0.47716418, -0.3372765),
              T(-0.4738406, -0.09567301, -0.21502851, 0.07263356),
              T(0.21500742, -0.09957578, 0.05073479, 0.5063499),
              T(-0.95140356, -0.19597691, 0.3108005, 0.3067237))).t())
      }
    }

    paramsTable = attention2.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_q") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.09555588, 0.16374706, -0.81079763, 0.18353464),
            T(0.72976017, -0.6785369, -0.1633139, -0.1220759),
            T(-0.47357813, 0.19808318, 0.63312566, -0.14370666),
            T( 0.11398887, 0.7884044, -0.36504376, -0.17514746))).t())
      } else if (i.toString contains "_k") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T(-0.19676681, -0.24631989, -1.1253904, -0.2751462),
            T(-0.17718858, 0.06754616, 0.5731753, -0.8507766),
            T( 0.06555229, -0.04867446, -0.05025194, -0.5535116),
            T(-0.5346166, 0.23926297, -0.4628236, -0.3947385))).t())
      } else if (i.toString contains "_v") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(Tensor[Float](
          T(T( 0.92687607, -0.545517, -0.05255984, 0.28678837),
            T( 0.34195843, 0.3929567, 0.51847, 0.7892322),
            T( 0.90397906, -0.9298378, 0.8783962, 0.2852646),
            T( 0.6237778, 0.3783044, 0.37894192, 0.42552295))).t())
      } else if (i.toString contains "_output_transform") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-1.9982174e-01, 1.4843611e-01, 4.4536388e-01, -3.4881935e-01),
              T(6.5677509e-02, 7.3198605e-01, 4.1394565e-01, 3.6246496e-01),
              T(3.8297844e-01, -2.0218496e-01, -6.0479283e-01, -8.4035518e-04),
              T(8.8539845e-01, 8.1015706e-02, -2.0919992e-01, -3.2815292e-01))).t())
      }
    }
    paramsTable = ffn1.getParametersTable()
    for (i <- paramsTable.keySet) {
      if (i.toString contains "_filter_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.3522124, -0.51549995, -0.67411846, 0.27011815, 0.6126283, -0.5052634,
               0.88756555, 0.47037336),
            T( 0.15704805, -0.11248052, 0.45173776, 1.0609885, -0.02032901, -0.272949,
              -0.27566594, 0.45384774),
            T( 0.6470523, -0.6543102, -0.21736439, -0.43480754, -0.13311917, -1.1141537,
              -0.59988606, -0.24346256),
            T( 0.11163724, -0.03015788, 0.38666677, -0.39999688, -0.53780854, -0.09386043,
              -0.09019023, 0.28964663))).transpose(1, 2))
      } else if (i.toString contains "_output_layer") {
        paramsTable.get[Table](i).get[Tensor[Float]]("weight").copy(
          Tensor[Float](
            T(T(-0.28514335, -0.5174819, -0.3048153, 0.16713372),
              T(-0.2276286, -0.31804547, 0.269992, 0.03182783),
              T(-0.26096576, -0.49425197, -0.23944728, 0.28338984),
              T( 0.260591, -0.17206982, -0.14490226, -0.20425473),
              T( 0.38700444, -0.5851576, 0.309289, -0.28129402),
              T(-0.03296154, -0.47809625, 0.43516076, 0.21953852),
              T(-0.38866428, 0.52283365, -0.60793763, 0.33401495),
              T(-0.29918984, 0.6243824, -0.21915461, -0.14608558))).transpose(1, 2))
      }
    }

    val expectedOutput = Tensor[Float](
      T(T(T(1.5693761, -1.0056276, 0.14640914, -0.71015745),
        T(1.4049922, -1.1252292, 0.46041852, -0.74018157),
        T(-0.7806267, -0.13584259, -0.75671536, 1.6731848),
        T(-0.3983218, -0.9217702, -0.36959055, 1.6896812),
        T(-0.62736577, -1.1783588, 0.36084852, 1.4448758),
        T(-0.29645187, -1.3115996, 0.1336132, 1.4744384)),
        T(T(1.281556, -1.111587, 0.65917075, -0.82913977),
          T(1.3174573, -1.1678243, 0.59200275, -0.74163586),
          T(0.68878394, -0.01719818, -1.6202699, 0.9486842),
          T(1.706251, -0.6772593, -0.29021385, -0.738778),
          T(-0.47597468, -0.88766754, -0.33201644, 1.6956586),
          T(-0.82912564, -0.8601543, 0.08772265, 1.6015573)))
    )

    val input1 = Tensor[Float](T(T(3, 1, 2, 3, 4, 5), T(6, 7, 8, 9, 10, 11))).add(1.0f)
    val input2 = Tensor[Float](T(T(4, 5, 7, 9, 10, 11), T(4, 12, 6, 3, 2, 15))).add(1.0f)
    val output = transformer.forward(T(input1, input2))
    output should be(expectedOutput)

    val gradInput = transformer.backward(T(input1, input2), output)

    println("done")
  }

  "AttentionBiasConstant" should "work correctly" in {
    val layer = new EncodePositionConstant[Float]()

    val input = Tensor[Float](T(T(
      T(1.5575712, 1.6023955, 1.4487493, 0.46178865),
      T(1.4542825, 0.36078143, 1.0112681, 1.7850459),
      T(1.0922418, 1.8467345, 0.17114377, 1.5875602),
      T(1.3181713, 1.1110513, 0.31925488, 0.61749554),
      T(0.30953693, 0.93909645, 1.9877799, 1.2225482),
      T(1.3529022, 0.3599646, 1.3499286, 0.4491992)),
      T(T(0.10186243, 0.9201369, 1.6568646, 0.47073865),
        T(1.950448, 1.6722536, 0.5169549, 0.83770823),
        T(1.4055192, 1.535857, 1.0745583, 1.4468269),
        T(0.53809, 0.01234245, 0.06532454, 0.1288917),
        T(1.6856189, 1.4987106, 0.1509037, 1.2490149),
        T(0.6981592, 1.1585901, 1.1459568, 0.3643551))))
    val output = layer.forward(input)

    val outputExpected = Tensor[Float](
      T(T( 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 1.0000000e+00),
        T( 8.4147096e-01, 9.9999990e-05, 5.4030228e-01, 1.0000000e+00),
        T( 9.0929741e-01, 1.9999998e-04, -4.1614681e-01, 1.0000000e+00),
        T( 1.4112000e-01, 2.9999996e-04, -9.8999250e-01, 9.9999994e-01),
        T(-7.5680250e-01, 3.9999996e-04, -6.5364361e-01, 9.9999994e-01),
        T(-9.5892429e-01, 4.9999997e-04, 2.8366220e-01, 9.9999988e-01))
    )

    output should be(outputExpected)
  }

  "transformer prepare decode layer" should "work correctly" in {
    val prepare = new TransformerPrepareDecoder[Float]()

    val input = Tensor[Float](
        T(T(T( 16.24345364, -6.11756414, -5.28171752, -10.72968622),
        T(8.65407629, -23.01538697, 17.44811764, -7.61206901),
        T(3.19039096, -2.49370375, 14.62107937, -20.60140709),
        T(-3.22417204, -3.84054355, 11.33769442, -10.99891267),
        T(-1.72428208, -8.77858418, 0.42213747, 5.82815214),
        T(-11.00619177, 11.4472371, 9.01590721, 5.02494339)),
        T(T(9.00855949, -6.83727859, -1.22890226, -9.35769434),
        T(-2.6788808, 5.30355467, -6.91660752, -3.96753527),
        T(-6.871727, -8.45205641, -6.71246131, -0.12664599),
        T(-11.17310349, 2.34415698, 16.59802177, 7.42044161),
        T(-1.91835552, -8.87628964, -7.47158294, 16.92454601),
        T(0.50807755, -6.36995647, 1.90915485, 21.00255136))))

    val expectedOutput = Tensor[Float](
        T(T(T(0, 0, 1, 1),
        T(17.084925, -6.117464, -4.741415, -9.729686),
        T(9.563374, -23.015186, 17.031971, -6.612069),
        T(3.331511, -2.493404, 13.631087, -19.601408),
        T(-3.9809747, -3.8401434, 10.684051, -9.998913),
        T(-2.6832063, -8.778085, 0.7057997, 6.828152)),
        T(T(0, 0, 1, 1),
        T(9.85003, -6.837178, -0.68859994, -8.357695),
        T(-1.7695832, 5.3037543, -7.332754, -2.9675353),
        T(-6.730607, -8.4517565, -7.702454, 0.87335396),
        T(-11.929906, 2.344557, 15.944379, 8.420442),
        T(-2.8772798, -8.87579, -7.1879206, 17.924545))))

    val expectedGradInput = Tensor[Float](
      T(T(T(17.084925, -6.117464, -4.741415, -9.729686),
        T(9.563374, -23.015186, 17.031971, -6.612069),
        T(3.331511, -2.493404, 13.631087, -19.601408),
        T(-3.9809747, -3.8401434, 10.684051, -9.998913),
        T(-2.6832063, -8.778085, 0.7057997, 6.828152),
        T(0, 0, 0, 0)),
      T(T(9.85003, -6.837178, -0.68859994, -8.357695),
        T(-1.7695832, 5.3037543, -7.332754, -2.9675353),
        T(-6.730607, -8.4517565, -7.702454, 0.87335396),
        T(-11.929906, 2.344557, 15.944379, 8.420442),
        T(-2.8772798, -8.87579, -7.1879206, 17.924545),
        T(0, 0, 0, 0))))

    val out = prepare.forward(input)
    out should be(expectedOutput)

    val out2 = prepare.backward(input, out)
    out2 should be(expectedGradInput)

  }

  "SelfAttentionBiasConstant layer" should "work correctly" in {
    val prepare = new SelfAttentionBiasConstant[Float]()
    val input = Tensor[Float](T(T(
        T( 16.24345364, -6.11756414, -5.28171752, -10.72968622),
        T(  8.65407629, -23.01538697, 17.44811764, -7.61206901),
        T(  3.19039096, -2.49370375, 14.62107937, -20.60140709),
        T( -3.22417204, -3.84054355, 11.33769442, -10.99891267),
        T( -1.72428208, -8.77858418, 0.42213747, 5.82815214),
        T(-11.00619177, 11.4472371, 9.01590721, 5.02494339)),
        T(T(  9.00855949, -6.83727859, -1.22890226, -9.35769434),
        T( -2.6788808, 5.30355467, -6.91660752, -3.96753527),
        T( -6.871727, -8.45205641, -6.71246131, -0.12664599),
        T(-11.17310349, 2.34415698, 16.59802177, 7.42044161),
        T( -1.91835552, -8.87628964, -7.47158294, 16.92454601),
        T(  0.50807755, -6.36995647, 1.90915485, 21.00255136))))

    val expectedOutput = Tensor[Float](
      T(T(T(T(0.0f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, -1e9f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, -1e9f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, -1e9f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1e9f),
      T(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)))))

    val out = prepare.forward(input)
    out should be(expectedOutput)

    val out2 = prepare.backward(input, out)
  }

  "TransformerOperation getPaddingBias" should "work good" in {
    val input = Tensor[Float](T(0, 1, 2, 3, 4, 5, 6, 7)).resize(Array(2, 4))
    val ops = TransformerOperation.getPaddingBias(input)
    val opsExpected = Tensor[Float](T(1.0f, 0.0f, 0f, 0f, 0f, 0f, 0f, 0f))
      .resize(Array(2, 1, 1, 4))
    ops should be(opsExpected)
  }

}

class TransformerConstantSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new SelfAttentionBiasConstant[Float]().setName("TransformerConstant")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class TransformerPrepareDecoderSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = new TransformerPrepareDecoder[Float]().setName("TransformerPrepareDecoder")
    val input = Tensor[Float](2, 6, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input)
  }
}

class TransformerLayerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val hiddenSize = 4
    val numHeads = 2
    val filterSize = 3
    val num_hidden_layers = 2
    val postprocessDropout = 1.0f
    val attentionDropout = 1.0f
    val reluDropout = 1.0f
    val model = new TransformerLayer[Float](20,
      hiddenSize, numHeads, filterSize, num_hidden_layers,
      postprocessDropout, attentionDropout, reluDropout)
    val input = Tensor[Float](2, 6).apply1(_ => Random.nextInt(10) + 1)
    runSerializationTest(model, input)
  }
}
