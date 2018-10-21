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

import java.nio.ByteOrder
import java.util.List

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.NodeType.TensorFlowNode
import com.intel.analytics.bigdl.nn.{GeneralGraph, Graph, StaticGraph, mkldnn}
import com.intel.analytics.bigdl.nn.mkldnn.{MklDnnModule, TFtoDnn}
import com.intel.analytics.bigdl.nn.tf.AssignGrad
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node}
import com.intel.analytics.bigdl.utils.tf.{Context, TensorflowLoader}
import org.tensorflow.framework.NodeDef

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class TFtoDnnConverter[T: ClassTag](defGraph: GeneralGraph[T, NodeDef])
  (implicit ev: TensorNumeric[T]) extends DnnLayerConverter[T, NodeDef]{

  mappingSet = init()

  private def init(): mutable.HashMap[String, NodeDef => MklDnnModule] = {
    val set = new mutable.HashMap[String, NodeDef => MklDnnModule]
    set("RELU") = this.toRelu
    set("DROPOUT") = this.toDropout
    set
  }

  override def toDnn(nodes: List[Node[NodeDef]]): Boolean = {
    var i = 0
    var enableDnn = true
    while (i < nodes.toArray.length) {
      val e = nodes.get(i)
      if (e.element != null) {
        if (!mappingSet.contains(e.element.getOp())) {
          enableDnn = false
          println(e.element.getOp)
        }
      }
      i += 1
    }
    enableDnn
  }

  override def convert(): Graph[T] = {
    val nodes = defGraph.nodes()
    val realInputNames = defGraph.inputs
      .map(i => if (i.split(":").length == 2) i.split(":")(0) else i)
      .distinct

    // Construct tf node graph
    val (tfGraph, newInputMap, _) =
    TensorflowLoader.buildToDirectedGraph(nodes, defGraph.outputs,
      (node: NodeDef) => realInputNames.contains(node.getName))

    buildDnnModel[T](tfGraph, realInputNames, defGraph.outputs, ByteOrder.LITTLE_ENDIAN)
  }

  private def buildDnnModel[T: ClassTag](
     tfGraph: DirectedGraph[NodeDef],
     inputs: Seq[String],
     outputs: Seq[String],
     byteOrder: ByteOrder,
     graphPrototxt: String = "",
     ctx: Option[Context[T]] = None,
     generatedBackward: Boolean = true
   )(implicit ev: TensorNumeric[T]): Graph[T] = {
    import scala.collection.JavaConverters._

    // Map from tensorflow node to the converted BigDL node
    val convertedNode =
      new mutable.HashMap[Node[NodeDef], Node[AbstractModule[Activity, Activity, T]]]()
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
      if (n.element == null || convertedNode.get(n).isDefined) {
        // Dummy node or converted node, skip
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
            case e: Throwable => throw new UnsupportedOperationException(errorMsg, e)
          }

//        // set name
//        if (nodes.size() == 1) {
//          // Use tf operation name if one to one map
//          module.setName(TensorflowLoader.removeColon(nodes.get(0).element.getName()))
//        } else {
//          // Many to one map
//          val name = TensorflowLoader.removeColon(TensorflowLoader.findCommonPrefix(nodes.asScala.map(_.element.getName)))
//          if (name == "") {
//            // Use a name combine nodes
//            module.setName(s"[${
//              nodes.asScala.map(_.element.getName).map(_.replaceAll("/", "\\\\"))
//                .map(TensorflowLoader.removeColon(_)).mkString(", ")
//            }]")
//          } else {
//            // Use the common name
//            module.setName(name + "/" + module.getName())
//          }
//        }
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

    val outputModules = tfGraph.source.prevNodes.map(_.element.getName).map(nameToNode)

    // todo: ????
    // connect(outputModules)

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
