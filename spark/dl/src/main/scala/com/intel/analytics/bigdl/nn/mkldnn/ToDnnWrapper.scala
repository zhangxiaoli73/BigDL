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

package com.intel.analytics.bigdl.nn.mkldnn

import java.nio.ByteOrder

import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat.NCHW
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn.tf._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}
import com.intel.analytics.bigdl.utils.tf.{Context, TensorflowLoader}
import com.intel.analytics.bigdl.utils.tf.loaders.Utils._
import org.tensorflow.framework.NodeDef
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.mkldnn.NodeType._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.tf.loaders.TensorflowOpsLoader

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import java.util.List

sealed class NodeType

object NodeType {
  case object CaffeNode extends NodeType

  case object TensorFlowNode extends NodeType

  case object DnnNode extends NodeType

  case object BigDLNode extends NodeType

  case object NullNode extends NodeType
}

object ToGraphWrapper {

  private var nodeType: NodeType = BigDLNode // default

  private def getNodeType[T: ClassTag, D](node: Node[D])
    (implicit ev: TensorNumeric[T]): NodeType = {
    if (node.element == null) {
      NullNode
    } else if (node.element.isInstanceOf[NodeDef]) {
      TensorFlowNode
    } else if (node.element.isInstanceOf[MklDnnLayer]) {
      DnnNode
      // } else if (node.element.isInstanceOf[TensorModule]) {
      // todo: may should be TensorModule
    } else {
      throw new Exception("node type not supported")
    }
  }

  def apply[T: ClassTag, D: ClassTag](generalGraph: GeneralGraph[T, D])
    (implicit ev: TensorNumeric[T]): Graph[T] = {

    import java.util.List

    val nodes = generalGraph.nodes()
    // just whether convert to dnn or bigdl
    nodeType = getNodeType[T, D](nodes.get(1))

    if (nodeType == TensorFlowNode) {
      val nodes = generalGraph.nodes().asInstanceOf[List[Node[NodeDef]]]
      if (TFtoDnn.enable(nodes.asInstanceOf[List[Node[NodeDef]]])) {
        val realInputNames = generalGraph.inputs
          .map(i => if (i.split(":").length == 2) i.split(":")(0) else i)
          .distinct

        // Construct tf node graph
        val (tfGraph, newInputMap, _) =
        TensorflowLoader.buildToDirectedGraph(nodes, generalGraph.outputs,
          (node: NodeDef) => realInputNames.contains(node.getName))

        val m = TFtoDnn.buildDnnModel[T](tfGraph,
          realInputNames, generalGraph.outputs, ByteOrder.LITTLE_ENDIAN)

        val t = 0
      }
    }

    val input = nn.Input[T]()
    val output = nn.ReLU[T]().inputs(input)
    Graph(input, output)
  }
}

object TFtoDnn {


  private val nodeSet = Array(
      "Add", "Relu", "Const", "BiasAdd",
      "MatMul", "Tanh", "Identity", "Placeholder")

  private var enableDnn : Boolean = false

  def enable(nodes: List[Node[NodeDef]]): Boolean = {
    enableDnn = true

    var i = 0


    while (i < nodes.toArray.length) {
      val e = nodes.get(i)
      if (e.element != null) {
        if (!nodeSet.contains(e.element.getOp())) {
          enableDnn = false
          println(e.element.getOp)
        }
      }
      i += 1
    }
//
//    nodes.toArray.foreach(n => {
//      if (n.element != null) {
//        val e = n.element.asInstanceOf[NodeDef]
//        if (!nodeSet.contains(e.getOp())) {
//          enableDnn = false
//          println(e.getOp)
//        }
//      }
//    })
    enableDnn
  }

  def apply[T: ClassTag](
   node: Node[NodeDef],
   byteOrder: ByteOrder,
   context: Context[T])(implicit ev: TensorNumeric[T])
   : TFtoDnn[T] = new TFtoDnn[T](node, byteOrder, context)

//  def apply[T: ClassTag](generalGraph: GeneralGraph[T, NodeDef])(implicit ev: TensorNumeric[T])
//    : StaticGraph[T] = {
//
//  }
//
//  private[bigdl] def toDirectGraph[T](
//     nodes: List[Node[NodeDef]],
//     inputs: Seq[String],
//     outputs: Seq[String],
//     byteOrder: ByteOrder,
//     graphPrototxt: String
//   )(implicit ev: TensorNumeric[T]): Module[T] = {
//
//
//
//  }

  private[bigdl] def buildDnnModel[T: ClassTag](
     tfGraph: DirectedGraph[NodeDef],
     inputs: Seq[String],
     outputs: Seq[String],
     byteOrder: ByteOrder,
     graphPrototxt: String = "",
     ctx: Option[Context[T]] = None,
     generatedBackward: Boolean = true
   )(implicit ev: TensorNumeric[T]): Module[T] = {
    import scala.collection.JavaConverters._

    // Map from tensorflow node to the converted BigDL node
    val convertedNode = new mutable.HashMap[Node[NodeDef],
      Node[AbstractModule[Activity, Activity, T]]]()
    val nameToNode =
      new mutable.HashMap[String, Node[AbstractModule[Activity, Activity, T]]]()

    val moduleToInputNodes =
      new mutable.HashMap[Node[AbstractModule[Activity, Activity, T]], Seq[Node[NodeDef]]]()
    val moduleToAllNodes =
      new mutable.HashMap[Node[AbstractModule[Activity, Activity, T]], Set[Node[NodeDef]]]()
    val context = ctx.getOrElse(new Context[T])

    // BFS to keep the input order same
    val bfsGraph = tfGraph.BFS
    bfsGraph.foreach(n => {
      if (n.element == null) {
        // Dummy node, skip
      } else if (convertedNode.get(n).isDefined) {
        // converted node, skip
      } else {
        val errorMsg =
          s"""
             | Cannot convert the given tensorflow operation graph to BigDL model. The convert fails
             | at node ${n.element.getName}. Operation type is ${n.element.getOp}
          """.stripMargin

        val (module, nodes, inputNodes) =
          try {
            val m = new TFtoDnn(n, byteOrder, null)
            (m.toDnn(), Seq(n).asJava, Seq(n))
          } catch {
            case e: Throwable =>
              throw new UnsupportedOperationException(errorMsg, e)
          }

        // set name
        if (nodes.size() == 1) {
          // Use tf operation name if one to one map
          module.setName(TensorflowLoader.removeColon(nodes.get(0).element.getName()))
        } else {
          // Many to one map
          val name = TensorflowLoader.removeColon(TensorflowLoader.findCommonPrefix(nodes.asScala.map(_.element.getName)))
          if (name == "") {
            // Use a name combine nodes
            module.setName(s"[${
              nodes.asScala.map(_.element.getName).map(_.replaceAll("/", "\\\\"))
                .map(TensorflowLoader.removeColon(_)).mkString(", ")
            }]")
          } else {
            // Use the common name
            module.setName(name + "/" + module.getName())
          }
        }
        // no switch
        //        val node = module match {
        //          case _: SwitchOps[_] => new SwitchControlNode(module)
        //          case _ => Node(module)
        //        }

        val node = Node(module)

        nodes.asScala.foreach(m => {
          convertedNode(m) = node
          nameToNode(m.element.getName) = node
        })

        moduleToInputNodes(node) = inputNodes
        moduleToAllNodes(node) = nodes.asScala.toSet
      }
    })

    /**
      * Go through all tensorflow nodes
      *
      * @param outputModuleNode
      */
    def connect(outputModuleNode: Seq[Node[AbstractModule[Activity, Activity, T]]]) = {
      val queue = new mutable.Queue[Node[AbstractModule[Activity, Activity, T]]]()
      val visited = mutable.Set[Node[AbstractModule[Activity, Activity, T]]]()
      queue.enqueue(outputModuleNode: _*)

      while (queue.nonEmpty) {
        val currNode = queue.dequeue()
        if (!visited(currNode)) {
          visited += currNode
          val inputNodes = moduleToInputNodes(currNode)
          val allNodes = moduleToAllNodes(currNode)
          val inputModuleNodes = inputNodes.flatMap(_.prevNodesAndEdges)
            .filterNot(n => context.containsTensor(n._1.element.getName) &&
              n._1.element.getOp() != "VariableV2")
            .filterNot(n => allNodes(n._1))
            .map(n => (convertedNode(n._1), n._2.newInstance())).filter(n => n._1 != currNode)
          inputModuleNodes.foreach(n => n._1.add(currNode, n._2))
          queue.enqueue(inputModuleNodes.map(_._1): _*)
        }
      }
    }

    val outputModules = tfGraph.source.prevNodes.map(_.element.getName).map(nameToNode)

    connect(outputModules)

    val inputNodes = inputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))
    val outputNodes = outputs
      .map(n => nameToNode.getOrElse(n, throw new IllegalArgumentException(s"Can't find node $n")))


    val weights = ArrayBuffer[Tensor[T]]()
    val gradients = ArrayBuffer[Tensor[T]]()
    for ((weight, grad, _) <- context.tensors) {
      weights += weight
      gradients += grad
    }

    // Append assign nodes
    val adjustOutputs = if (context.assignGrads.isDefined) {
      outputNodes.map(n => {
        val matchNode = context.assignGrads.get.filter(_._2 == n.element.getName())
        require(matchNode.size <= 1, "Invalid gradients output")
        if (matchNode.size == 1) {
          new AssignGrad[T](context(matchNode.head._1)._2).inputs(n)
        } else {
          n
        }
      })
    } else {
      outputNodes
    }

    // convert to dnnGraph
    Graph.dynamic(inputNodes.toArray, adjustOutputs.toArray,
      Some((weights.toArray, gradients.toArray)),
      generatedBackward)
  }
}

class TFtoDnn[T: ClassTag](node: Node[NodeDef], byteOrder: ByteOrder,
                   context: Context[T]) (implicit ev: TensorNumeric[T]) {

  private val nodeDef = node.element

  def toDnn(): AbstractModule[Activity, Activity, T] = {
    nodeDef.getOp() match {
      case "Relu" => toRelu()
      case "Identity" => toIdentity()
      case "Const" => toConst()
      case _ => throw new Exception("not supported")
    }
  }

  def toRelu(): AbstractModule[Activity, Activity, T]
    = ReLU().asInstanceOf[AbstractModule[Activity, Activity, T]]

  def toMaxPooling(): AbstractModule[Activity, Activity, T] = {
    val attributes = nodeDef.getAttrMap
    val format = getString(attributes, "data_format")
    val strideList = getIntList(attributes, "strides")
    val kernelList = getIntList(attributes, "ksize")

    require(format == NCHW, "dnn maxpooling just support nchw")
    val (strideH, strideW, ksizeH, ksizeW) = format match {
      case "NHWC" =>
        require(strideList(3) == 1, s"not support strides on depth")
        (strideList(1), strideList(2), kernelList(1), kernelList(2))
      case "NCHW" =>
        require(strideList(1) == 1, s"not support strides on depth")
        (strideList(2), strideList(3), kernelList(2), kernelList(3))
      case _ =>
        throw new IllegalArgumentException(s"not supported data format: $format")
    }

    val (pW, pH) =
      if (getString(attributes, "padding") == "SAME") {
        (-1, -1)
      } else {
        (0, 0)
      }

//    SpatialMaxPooling[T](ksizeW, ksizeH, strideW, strideH, pW, pH,
//      format = DataFormat(format))
    MaxPooling(ksizeW, ksizeH, strideW, strideH, pW, pH)
      .asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  def toLRN(): AbstractModule[Activity, Activity, T] = {
    val size = utils.tf.loaders.Utils.getInt(nodeDef.getAttrMap, "depth_radius")
    val k = utils.tf.loaders.Utils.getFloat(nodeDef.getAttrMap, "bias")
    val alpha = utils.tf.loaders.Utils.getFloat(nodeDef.getAttrMap, "alpha")
    val beta = utils.tf.loaders.Utils.getFloat(nodeDef.getAttrMap, "beta")
    LRN(
      size = size * 2 + 1,
      k = k,
      alpha = alpha * (size * 2 + 1),
      beta = beta
    ).asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  def toCAddTable(): AbstractModule[Activity, Activity, T] = {
    CAddTable().asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  def toShape(): AbstractModule[Activity, Activity, T] = {
    com.intel.analytics.bigdl.nn.tf.Shape[T]()
  }

  def toLinear(): AbstractModule[Activity, Activity, T] = {
    // todo:
    com.intel.analytics.bigdl.nn.tf.Shape[T]()
  }

  def toDropout(): AbstractModule[Activity, Activity, T] = {
    // todo:
    com.intel.analytics.bigdl.nn.tf.Shape[T]()
  }

  def toConv(): AbstractModule[Activity, Activity, T] = {
    // todo:
    com.intel.analytics.bigdl.nn.tf.Shape[T]()
  }

  def toIdentity(): AbstractModule[Activity, Activity, T] = {
    // todo:
    com.intel.analytics.bigdl.nn.tf.Shape[T]()
  }

  def toConst(): AbstractModule[Activity, Activity, T] = {
    // todo:
    com.intel.analytics.bigdl.nn.tf.Shape[T]()
  }
}

class BigDLtoDnn[T: ClassTag](nodeDef: Module[T]) (implicit ev: TensorNumeric[T]) {

}