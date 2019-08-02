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

import breeze.linalg.{*, dim}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.{T, Table}
import com.sun.org.apache.xpath.internal.operations.Bool
import org.apache.spark.sql.catalyst.expressions.If
import org.dmg.pmml.{Apply, False}

class FPNPostProcessor(
    score_thresh: Float,
    nms_thresh: Float,
    detections_per_img: Int,
    cls_agnostic_bbox_reg: Boolean,
    bbox_aug_enabled: Boolean
  ) (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {

  val softMax = SoftMax[Float]()

  val weight = Tensor[Float](T(1.0f, 1.0f, 1.0f, 1.0f))

  /**
    *   """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
  """
    * @param num_classes
    */
  def filterResults(boxes: Tensor[Float], scores: Tensor[Float], num_classes: Int,
                    imgInfo: Tensor[Float]): Array[RoiLabel] = {
    val dim = num_classes * 4
    boxes.resize(Array(boxes.nElement() / dim, dim))
    scores.resize(Array(scores.nElement() / num_classes, num_classes))

    // check scores >
    val results = new Array[RoiLabel](num_classes)
    var clsInd = 1
    while (clsInd < num_classes) {
      results(clsInd) = postProcessOneClass(scores, boxes, clsInd)
      clsInd += 1
    }

    results
  }

    var nmsTool: Nms = new Nms
    private var areas: Tensor[Float] = _

    def postProcessOneClass(scores: Tensor[Float], boxes: Tensor[Float],
                            clsInd: Int): RoiLabel = {
      val inds = (1 to scores.size(1)).filter(ind =>
        scores.valueAt(ind, clsInd + 1) > score_thresh).toArray
      if (inds.length == 0) return null
      val clsScores = selectTensor(scores.select(2, clsInd + 1), inds, 1)
      val clsBoxes = selectTensor(boxes.narrow(2, clsInd * 4 + 1, 4), inds, 1)

      val keepN = nmsTool.nms(clsScores, clsBoxes, nms_thresh, inds)

      val bboxNms = selectTensor(clsBoxes, inds, 1, keepN)
      val scoresNms = selectTensor(clsScores, inds, 1, keepN)

      RoiLabel(scoresNms, bboxNms)
    }

    private def selectTensor(matrix: Tensor[Float], indices: Array[Int],
      dim: Int, indiceLen: Int = -1, out: Tensor[Float] = null): Tensor[Float] = {
      assert(dim == 1 || dim == 2)
      var i = 1
      val n = if (indiceLen == -1) indices.length else indiceLen
      if (matrix.nDimension() == 1) {
        val res = if (out == null) {
          Tensor[Float](n)
        } else {
          out.resize(n)
        }
        while (i <= n) {
          res.update(i, matrix.valueAt(indices(i - 1)))
          i += 1
        }
        return res
      }
      // select rows
      if (dim == 1) {
        val res = if (out == null) {
          Tensor[Float](n, matrix.size(2))
        } else {
          out.resize(n, matrix.size(2))
        }
        while (i <= n) {
          res.update(i, matrix(indices(i - 1)))
          i += 1
        }
        res
      } else {
        val res = if (out == null) {
          Tensor[Float](matrix.size(1), n)
        } else {
          out.resize(matrix.size(1), n)
        }
        while (i <= n) {
          var rid = 1
          val value = matrix.select(2, indices(i - 1))
          while (rid <= res.size(1)) {
            res.setValue(rid, i, value.valueAt(rid))
            rid += 1
          }
          i += 1
        }
        res
      }
    }

  // todo: now just support one image per batch
  override def updateOutput(input: Table): Tensor[Float] = {
    val class_logits = input[Tensor[Float]](1)
    val box_regression = input[Tensor[Float]](2)
    val bbox = input[Tensor[Float]](3)

    val class_prob = softMax.forward(class_logits)
    val proposals = Tensor[Float]().resizeAs(box_regression).copy(box_regression)

    BboxUtil.decode(box_regression, bbox, weight, proposals)

    val num_classes = class_prob.size(1)
    val boxes_per_image = bbox.size(1)

    val proposals_split = proposals.split(boxes_per_image, dim = 1)
    val class_prob_split = class_prob.split(boxes_per_image, dim = 1)

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput
  }
}
