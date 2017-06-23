## ValidationMethod ##

ValidationMethod is a method to validate the model during model trainning or evaluation.
The trait can be extended by user-defined method. Now we have defined Top1Accuracy, Top5Accuracy, Loss.

#### Top1Accuracy ####

Caculate the percentage that output's max probability index equals target.

#### Top5Accuracy ####

Caculate the percentage that target in output's top5 probability indexes.

#### Loss ####

Calculate loss of output and target with criterion. The default criterion is ClassNLLCriterion.

#### Example code ####

Followings are examples to evaluate LeNet5 model with Top1Accuracy, Top5Accuracy, Loss validation method.

Scala example code:

```
val conf = Engine.createSparkConf()
val sc = new SparkContext(conf)
Engine.init
      
val data = new Array[Sample[Float]](10)
var i = 0
while (i < data.length) {
  val input = Tensor[Float](28, 28).fill(0.8f)
  val label = Tensor[Float](1).fill(1.0f)
  tmp(i) = Sample(input, label)
  i += 1
}
val model = LeNet5(classNum = 10)
val dataSet = sc.parallelize(data, 4)

val result = model.evaluate(dataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](), new Loss[Float]()))
```
result is

```

```

Python example code:
```
init_engine()
      
data_len = 10
batch_size = 2

def gen_rand_sample():
    features = np.random.uniform(0, 1, (FEATURES_DIM))
    label = (2 * features).sum() + 0.4
    return Sample.from_ndarray(features, label)

trainingData = self.sc.parallelize(range(0, data_len)).map(
    lambda i: gen_rand_sample())

model = build_model(10)    
test_results = trained_model.test(trainingData, batch_size, ["Top1Accuracy", "Top5Accuracy", "Loss"])
```

result is

```

```
