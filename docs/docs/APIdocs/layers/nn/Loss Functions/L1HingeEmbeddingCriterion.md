## L1HingeEmbeddingCriterion ##

**Scala:**
```scala
val model = L1HingeEmbeddingCriterion[T](margin)
```
**Python:**
```python
model = L1HingeEmbeddingCriterion(margin)
```

Creates a criterion that measures the loss given an input ``` x = {x1, x2} ```, a table of two Tensors, and a label y (1 or -1).
This is used for measuring whether two inputs are similar or dissimilar, using the L1 distance, and is typically used for learning nonlinear embeddings or semi-supervised learning.
```
             ⎧ ||x1 - x2||_1,                  if y ==  1
loss(x, y) = ⎨
             ⎩ max(0, margin - ||x1 - x2||_1), if y == -1
```
The margin has a default value of 1, or can be set in the constructor.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

val model = L1HingeEmbeddingCriterion[Float](0.6)
val input1 = Tensor[Float](2).rand()
val input2 = Tensor[Float](2).rand()
val input = T(input1, input2)
val target = Tensor[Float](1)
target(Array(1)) = 1.0f

val output = model.forward(input, target)
```
output is
```
output: Float = 0.84714425
```

**Python example:**
```python
model = L1HingeEmbeddingCriterion(0.6)
input1 = np.random.randn(2)
input2 = np.random.randn(2)
input = [input1, input2]
target = np.array([1.0])

output = model.forward(input, target)
```
output is
```
1.3705695
```
