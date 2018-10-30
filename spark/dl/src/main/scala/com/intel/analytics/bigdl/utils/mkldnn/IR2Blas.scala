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

package com.intel.analytics.bigdl.utils.mkldnn

import java.util

import com.intel.analytics.bigdl.nn.{DynamicGraph, Graph, StaticGraph, mkldnn}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import com.intel.analytics.bigdl.utils.caffe.CaffeConversionException
import com.intel.analytics.bigdl.utils.tf.loaders.TensorflowOpsLoader

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class IR2Blas[T: ClassTag](IRgraph: IRGraph[T])
  (implicit ev: TensorNumeric[T]) extends IRConverter[T](IRgraph) {

  private val IR2BlasMap = new mutable.HashMap[String, (IRElement) => Module[Float]]
  override def mapInit(): Unit = {
    IR2BlasMap("DropOut") = fromDropout
    IR2BlasMap("Identity") = fromIdentity
    IR2BlasMap("SpatialConvolution") = fromConv
    IR2BlasMap("MaxPool") = fromMaxPooling
    // todo: not complete
  }

  private var from : String = ""

  override def toGraph(): Graph[T] = {
    val nodes = IRgraph.allNodes
    nodes.foreach(n => {
      // from tensorflow
      if (n.element.isInstanceOf[TFElement]) {
        from = "tensorflow"
      }
    })
    require(from == "tensorflow", "only support convert tensorflow to blas")
    tf2Graph()
  }

  private def tf2Graph() : Graph[T] = {
    val nodes = IRgraph.allNodes
    val oldToNew = new util.HashMap[Node[IRElement], Node[Module[T]]]()
    nodes.foreach(node => {
      val blas = if (IR2BlasMap.contains(node.element.getOp())) {
        IR2BlasMap(node.element.getOp())(node.element)
      } else {
        require(node.element.isInstanceOf[TFElement], "graph element should be tensorflow element")
        try {
          val cls = Class.forName("com.intel.analytics.bigdl.utils.tf.loaders." +
            node.element.getOp)
          val builder = cls.getConstructors()(0).newInstance().asInstanceOf[TensorflowOpsLoader]
          builder.build[T](node.element.asInstanceOf[TFElement].getLayer(), null, null)
        } catch {
          case e: Throwable =>
            throw new UnsupportedOperationException(s"unsupported op ${node.element.getOp()}", e)
        }
      }
      oldToNew.put(node, new Node(blas.asInstanceOf[Module[T]]))
    })

    nodes.foreach(node => {
      node.nextNodesAndEdges.foreach(nextNodeAndEdge => {
        if (oldToNew.containsKey(nextNodeAndEdge._1)) {
          oldToNew.get(node).add(oldToNew.get(nextNodeAndEdge._1), nextNodeAndEdge._2)
        }
      })
    })

    val inputs = IRgraph.inputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[T]])
    val outputs = IRgraph.outputs.toArray.map(n => oldToNew.get(n).asInstanceOf[ModuleNode[T]])

    new DynamicGraph[T](inputs, outputs, IRgraph.variables, IRgraph.generateBackward)
  }

  override def enableConvert(): Boolean = {
   true
  }

  private def fromConv(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromMaxPooling(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromSelectTable(node: IRElement) : Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromSpatialBatchNormalization(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromTranspose(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromContiguous(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromTemporalConvolution(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromLinear(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromDropout(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

  private def fromIdentity(node: IRElement): Module[Float] =
    throw new UnsupportedOperationException("not implement")

}

object IR2Blas {
  def apply[T: ClassTag](IRgraph: IRGraph[T])
    (implicit ev: TensorNumeric[T]): IR2Blas[T] = new IR2Blas(IRgraph)
}