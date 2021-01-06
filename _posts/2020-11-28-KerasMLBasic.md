---
layout: post
title: Keras机器学习的基础
---

# Keras机器学习的基础

## 概览
本课程包括关于使用Keras进行机器学习的基本概念的教程。
- [图像分类](#基本的图像分类)：使用fashingmist数据集进行图像分类。
- [回归](#基本的图像分类)：使用波士顿住房数据集进行回归。
- [文本分类](#基本的图像分类)：使用IMDB数据集进行文本分类。
- [过拟合和不拟合](#基本的图像分类)：学习ML中的这些重要概念。
- [保存和恢复](#基本的图像分类)：学习如何保存和恢复TensorFlow模型。

## 基本的图像分类

在本指南中，我们将训练一个神经网络模型来分类服装图像，如运动鞋和衬衫。如果您不理解所有的细节也没关系，这是一个完整的Keras程序的快速概述，详细信息将随我们的进展而解释。

```r
library(tensorflow)
library(keras)
```

### 导入Fashion MNIST数据集
```r 
fashion_mnist <- dataset_fashion_mnist()
```
本指南使用[Fashion MNIST]数据集，包含10个类别的7万张灰度图像。这些图片以低分辨率(28x28像素)展示了衣个别服，如下图所示:

```{r fig1, echo=FALSE, out.width="60%", fig.cap ='Fashion MNIST 样品(*Zalando, MIT License*)',fig.align='center'} 

knitr::include_graphics("https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/images/fashion-mnist-sprite.png")
```
[Fashion MNIST]的目的是替代经典的[MNIST]数据集，后者通常被用作计算机视觉机器学习程序的“Hello, World”。[MNIST]数据集包含手写数字(0、1、2等)的图像，其格式与我们将在这里使用的衣物数据相同。

本指南使用[Fashion MNIST]进行各种各样的操作，因为这是一个比常规[MNIST]更具挑战性的问题。这两个数据集都相对较小，用于验证算法是否如预期的那样工作。它们是测试和调试代码的良好起点。

我们将使用60,000张图像来训练网络，并使用10,000张图像来评估网络学习分类图像的准确性。你可以直接从Keras访问[Fashion MNIST]。

```r 
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
```

现在我们有四个数组: train_images和train_labels数组是训练集——即模型用来学习的数据。模型根据测试集进行测试的测试数据:test_images和test_labels。

每个图像都是28x28个数组，像素值在0到255之间。标签为整数数组，取值范围为0 ~ 9。这些对应于图像所代表的服装类别:
```{r tablable, echo=FALSE,warning=FALSE,message=FALSE} 
library("dplyr")
library("kableExtra") 
Digit<-c(0:9)
Class = c('T-shirt/top',              'Trouser',
                'Pullover',           'Dress',
                'Coat',               'Sandal',
                'Shirt',              'Sneaker',
                'Bag',                'Ankle boot')
funsiontab<-cbind(Digit,Class)
knitr::kable( funsiontab, caption = '服装类别及对应的数字编号',
    booktabs = TRUE, digits = '4', align='ccc',format.args = list(scientific = FALSE)) %>%
kable_paper("striped", full_width = F)  %>% 
kableExtra::kable_classic_2() 
```
每个图像都映射到单个标签。由于类名不包含在数据集中，所以我们将它们存储在一个向量中，以便稍后绘制图像时使用。
```r 
class_names = c('T-shirt/top',              'Trouser',
                'Pullover',           'Dress',
                'Coat',               'Sandal',
                'Shirt',              'Sneaker',
                'Bag',                'Ankle boot')
```

### 检视数据
在训练模型之前，让我们研究一下数据集的格式。如下图所示，训练集中有60000张图像，每张图像用28x28像素表示。

```r 
# 训练集中有60000张图像，每张图像用28x28像素表示。
dim(train_images)
dim(train_labels)
# 训练集每个标签是0到9之间的整数:
table(train_labels)

# 测试集中有10000张图像。同样，每张图像用28×28像素表示
dim(test_images)
dim(test_labels)
# 测试集每个标签是0到9之间的整数:
table(test_labels)

```

### 数据预处理

在训练网络之前，必须对数据进行预处理。如果你检查训练集中的第一张图像，你会看到像素值的范围是0到255:

```r 
library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")
```
```{r fig2, echo=FALSE,out.width="49%", fig.cap ='检查训练集中的第一张图像',fig.align='center'} 

knitr::include_graphics('https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_classification_files/figure-html/unnamed-chunk-9-1.png')
```
在输入到神经网络模型之前，我们将这些值缩放到0到1的范围内。对于这个，我们只需要除以255。重要的是训练集和测试集以相同的方式进行预处理:
```r 
train_images <- train_images / 255
test_images <- test_images / 255
```
显示训练集的前25张图像，并在每张图像下面显示类名。验证数据的格式是否正确，如果没问题，接下来就可以构建和训练模型了。
```r 
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}
```
```{r fig3, echo=FALSE,out.width="60%", fig.cap ='训练集的前25张图像和类名。',fig.align='center'} 

knitr::include_graphics('https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_classification_files/figure-html/unnamed-chunk-11-1.png')
```

### 构建模型

构建神经网络需要配置模型的层，然后编译模型。

#### 设置神经层

神经网络的基本构件是层(神经层)。层从输入到它们的数据中提取表征。并且，希望这些表征对于手头的问题更有意义。

大部分深度学习是将简单的层链接在一起构成的，其中大多数层（例如layer_dense）在训练模型时都有可以设定学习的参数。

```r 
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

```
该模型的第一层layer_flatten将图像格式从2维数组(28x28像素)转换为28*28 = 784像素的1维数组。可以把这一层想象成将图像中的像素行拆散并排列起来。这一层没有参数需要学习；它只是重新格式化数据。

在像素数据被单一化后，模型由两个密集层组成。这些是紧密相连或完全相连的神经层。第一密集层有128个节点(或神经元)。第二层(也是最后一层)是一个有10个节点的softmax层——它返回一个10个概率得分的数组，总和为1。每个节点都包含一个分数，该分数表示当前图像属于10个数字类之一的概率。

#### 编译模型
在模型准备好进行训练之前，还需要进行一些设置。这些是在模型的编译步骤中添加的:
- Loss函数(Loss function): 这度量了模型在训练期间的精确度。我们希望最小化这个函数，以“引导”模型朝正确的方向发展。
- 优化器(Optimizer ): 这是模型如何根据它看到的数据和它的损失函数进行更新的方式。
- 度量标准(Metrics): 用于监控培训和测试步骤。下面的示例使用准确度，即正确分类的图像的比例。

```r 
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
```

#### 训练模型

训练神经网络模型需要以下步骤:

- 将训练数据提供给模型——在本例中是train_images和train_labels数组。
- 这个模型学会了把图像和标签联系起来。
- 我们要求模型对测试集进行预测——在本例中是test_images数组。我们验证预测是否与test_labels数组中的标签相匹配。

要开始训练，调用fit方法-模型对训练数据进行“拟合(fit)”:
```r 
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)
## Train on 60000 samples
## Epoch 1/5
## 60000/60000 - 2s - loss: 0.4945 - accuracy: 0.8262
## Epoch 2/5
## 60000/60000 - 2s - loss: 0.3751 - accuracy: 0.8643
## Epoch 3/5
## 60000/60000 - 2s - loss: 0.3354 - accuracy: 0.8758
## Epoch 4/5
## 60000/60000 - 2s - loss: 0.3135 - accuracy: 0.8854
## Epoch 5/5
## 60000/60000 - 2s - loss: 0.2956 - accuracy: 0.8918
```
当模型运行时，损失和精度指标就会显示出来。该模型的精度约为0.8918(89.18%)。

#### 评估准确性

接下来，比较模型在测试数据集中的执行情况:

```r 
score <- model %>% evaluate(test_images, test_labels, verbose = 0)
score<-as.list(score)
cat('Test loss:', score$loss, "\n")
## Test loss: 0.3755946
cat('Test accuracy:', score$acc, "\n")
## Test accuracy: 0.8644
```
结果表明，测试数据集的精度(86.44%)略低于训练数据集的精度(89.18%)。训练精度和测试精度之间的差距就是**过拟合**的一个例子。过拟合是指机器学习模型在新数据上的表现比在训练数据上差。

#### 作出预测

经过训练的模型，我们可以用它来预测一些图像。
```r 
predictions <- model %>% predict(test_images)
```
这里，模型预测了测试集中每个图像的标签。让我们看看第一个预测:
```r 
predictions[1, ]
## [1] 5.465935e-06 1.288366e-07 3.570543e-06 1.659937e-08 2.075325e-05
## [6] 1.836076e-02 2.499909e-06 1.217376e-01 2.614871e-05 8.598431e-01
```
预测结果是一个由10个数字组成的数组。这些数值描述了模型判断该图像对应于10种不同的服装类型的“置信度”。 我们可以看到哪个标签的置信度最高：
```r 
which.max(predictions[1, ])
## [1] 10
```
由于标签(Labels)是基于0起始的，然而R语言的数据集标签是由1起始的，所以predictions[1, ]预测的标签为9。模型非常确信这张照片是一件踝靴(Ankle boot)。我们可以检查测试标签，看看预测结果是否正确。
```r 
test_labels[1]
```

或者，我们也可以直接得到类预测:
```r 
class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]
###  [1] 9 2 1 1 6 1 4 6 5 7 4 5 5 3 4 1 2 2 8 0
```

让我们用几幅图来说明模型的预测。正确的预测标签为绿色，错误的预测标签为红色。
```r 
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}
```
```{r fig4, echo=FALSE,out.width="60%", fig.cap ='检视部分模型预测结果，正确的预测标签为绿色，错误的预测标签为红色',fig.align='center'} 

knitr::include_graphics('https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_classification_files/figure-html/unnamed-chunk-21-1.png')
```

最后，利用训练好的模型对单个图像进行预测。

```r 
# 从测试数据集中获取一个图像
# 注意保持崎岖数据的维度信息，这是模型所期望的，利用drop = FALSE帮助关掉返回向量
str(test_images)
# num [1:10000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
img <- test_images[1, , , drop = FALSE]
dim(img)
# [1]  1 28 28
```
现在预测图像:
```r 
predictions <- model %>% predict(img)
predictions
##              [,1]         [,2]         [,3]         [,4]         [,5]
## [1,] 5.465944e-06 1.288367e-07 3.570535e-06 1.659934e-08 2.075324e-05
##            [,6]         [,7]      [,8]         [,9]    [,10]
## [1,] 0.01836077 2.499906e-06 0.1217377 2.614871e-05 0.859843

```
*predict*返回一个包含子列表的列表，每个子列表对应数据批中的某图像。在这里的批处理中获取我们的(唯一的)图像的预测:

```r 
# 因为标签是基于0的，所以减去1
prediction <- predictions[1, ] - 1
which.max(prediction)
# [1] 10

# 或者，直接再次获取类预测:
class_pred <- model %>% predict_classes(img)
class_pred
# [1] 9
```

[Fashion MNIST]:https://github.com/zalandoresearch/fashion-mnist
[MNIST]: http://yann.lecun.com/exdb/mnist/

## 回归

在回归问题中，我们的目标是预测一个连续值的输出，如价格或概率。与此形成对比的是分类问题，在分类问题中，我们的目标是预测一个离散的标签(例如，一张图片中包含一个苹果或橘子)。

本笔记建立了一个模型来预测20世纪70年代中期波士顿郊区房屋的中间价格。为此，我们将为模型提供一些关于郊区的数据点，如犯罪率和当地房产税率。

### 波士顿房价数据集

[波士顿房价]数据可以直接从keras获得。

```r 
library(keras)
library(tfdatasets)

boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

```

#### 实例和特点

这个数据集比我们目前使用的其他数据集要小得多:它总共有506个例子，分别在404个训练示例和102个测试示例之间划分:

```r 
paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))
## [1] "Training entries: 5252, labels: 404"
```

数据集包含13个不同的特征：

- 人均犯罪率。

- 超过25,000平方英尺的住宅用地比例。

- 每个城镇非零售业务英亩的比例。

- 查尔斯河虚拟变量（如果束缚河流，则为1；否则为0）。

- 一氧化氮浓度（千万分之一）。

- 每个住宅的平均房间数。

- 1940年之前建造的自有住房的比例。

- 到五个波士顿就业中心的加权距离。

- 径向公路的可达性指数。

- 每10,000美元的全值财产税率。

- 各镇的师生比例。

- 1000 *（Bk-0.63）** 2其中Bk是按城镇划分的黑人比例。

- 人口中处于较低地位的百分比。

输入数据的每个特性互相使用不同的标度存储。有些特征用0到1之间的比例表示，有些特征用1到12之间的范围表示，有些特征用0到100之间的范围表示，以此类推。
```r 
# 显示样品特征，注意不同的标度
train_data[1, ] 
##  [1]   1.23247   0.00000   8.14000   0.00000   0.53800   6.14200  91.70000
##  [8]   3.97690   4.00000 307.00000  21.00000 396.90000  18.72000
```
为数据添加列名，以便更好地检查数据。
```r 
library(dplyr)

column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

train_df <- train_data %>% 
  as_tibble(.name_repair = "minimal") %>% 
  setNames(column_names) %>% 
  mutate(label = train_labels)

test_df <- test_data %>% 
  as_tibble(.name_repair = "minimal") %>% 
  setNames(column_names) %>% 
  mutate(label = test_labels)
```

#### 标签

这些标签是的房价单位：千美元。
```r 
train_labels[1:10]
##  [1] 15.2 42.3 50.0 21.1 17.7 18.5 11.3 15.6 15.6 14.4
```

### 标准化特征数据

建议对使用不同标度和范围的特征数据进行标准化。虽然模型在没有特征归一化的情况下可能也会收敛，但这会使训练变得更加困难，并且会使得到的模型更加依赖于输入中使用的单元的选择。

我们将使用在*tfdatasets*包中实现的*feature_spec*接口进行标准化。*feature_columns*接口允许对表数据进行其他常见的预处理操作。
```r 
library(tfdatasets)

spec <- feature_spec(train_df, label ~ . ) %>% 
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
  fit()

spec
## ── Feature Spec ─────────────────────────────────────────────────────────────────────────── 
## A feature_spec with 13 steps.
## Fitted: TRUE 
## ── Steps ────────────────────────────────────────────────────────────────────────────────── 
## The feature_spec has 1 dense features.
## StepNumericColumn: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT 
## ── Dense features ─────────────────────────────────────────────────────────────────────────
```

使用*tfdatasets*创建的*spec*可以与*layer_dense_features*一起使用，直接在TensorFlow图中执行预处理。

我们可以看看这个*spec*创建的密集层的输出:

```r 
layer <- layer_dense_features(
  feature_columns = dense_features(spec), 
  dtype = tf$float32
)
layer(train_df)
```
注意，这将返回一个换算后值得的数据矩阵(在本例中说，它是一个二维的Tensor)。

#### 创建模型

接下来我们构建模型。这里我们将使用Keras functional API——这是使用feature_spec API时推荐的方式。注意，我们只需要从我们刚刚创建的*spec*中传递*dense_features*。

```r 
input <- layer_input_from_dataset(train_df %>% select(-label))

output <- input %>% 
  layer_dense_features(dense_features(spec)) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) 

model <- keras_model(input, output)

summary(model)
```

然后我们用以下方法编译模型:
```r 
model %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
```
我们将把模型构建代码包装成一个函数，以便能够在不同的实验中重用它。请记住，Keras *fit*会就地修改模型。
```r 
build_model <- function() {
  input <- layer_input_from_dataset(train_df %>% select(-label))
  
  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1) 
  
  model <- keras_model(input, output)
  
  model %>% 
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error")
    )
  
  model
}
```

#### 训练模型

对模型进行了500个epochs训练，并在keras_training_history对象中记录了训练和验证准确性。 我们还展示了如何使用自定义回调方法，将每个epochs的默认训练输出替换为一个点。
```r 
# 通过每完一个epochs打印一个点显示来训练进度。
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

model1 <- build_model()

history1 <- model1 %>% fit(
  x = train_df %>% select(-label),
  y = train_df$label,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)
```
现在，我们使用存储在*history*变量中的指标来可视化模型的训练进度。我们想用这些数据来确定在模型停止进步之前需要训练多久。

```r 
library(ggplot2)
plot(history1)
```
```{r fig5, echo=FALSE,out.width="60%", fig.cap ='训练模型的收敛过程和时间',fig.align='center'} 

knitr::include_graphics('https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_regression_files/figure-html/unnamed-chunk-13-1.png')
```
这张图表显示，在大约200个epochs之后，模型几乎没有什么改进。让我们更新*fit*方法，当验证分数没有提高时自动停止训练。我们将使用一个回调来测试每个epoch的训练条件。如果经过了一定数量的epoch，没有显示出改进，它会自动停止训练。
```r 
# patience parameter是要检查改进的时期数。
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model2 <- build_model()

history2 <- model2 %>% fit(
  x = train_df %>% select(-label),
  y = train_df$label,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop)
)

plot(history2)
#Error in data.frame(epoch = seq_len(x$params$epochs), value = unlist(values),  :
#  arguments imply differing number of rows: 500, 368, 2000

str(history2)
List of 2
# $ params :List of 3
#  ..$ verbose: int 0
#  ..$ epochs : int 500
#  ..$ steps  : int 11
# $ metrics:List of 4
#  ..$ loss                   : num [1:92] 494 398 309 226 157 ...
#  ..$ mean_absolute_error    : num [1:92] 20.4 17.9 15.4 12.7 10.3 ...
#  ..$ val_loss               : num [1:92] 504 410 319 227 169 ...
#  ..$ val_mean_absolute_error: num [1:92] 20.5 18.2 15.5 12.5 10.3 ...
# - attr(*, "class")= chr "keras_training_history"

history2$metrics$loss<-c(history2$metrics$loss,rep(NA,history2$params$epochs-length(history2$metrics$loss)))
history2$metrics$mean_absolute_error<-c(history2$metrics$mean_absolute_error,rep(NA,history2$params$epochs-length(history2$metrics$mean_absolute_error)))
history2$metrics$val_loss<-c(history2$metrics$val_loss,rep(NA,history2$params$epochs-length(history2$metrics$val_loss)))
history2$metrics$val_mean_absolute_error<-c(history2$metrics$val_mean_absolute_error,rep(NA,history2$params$epochs-length(history2$metrics$val_mean_absolute_error)))

plot(history2)
```
```{r fig6, echo=FALSE,out.width="60%", fig.cap ='训练模型的收敛过程和时间2',fig.align='center'} 

knitr::include_graphics('https://s3.ax1x.com/2020/12/25/rWN4UK.png')
```
该图显示平均误差约为2500美元。 这个好吗？ 好吧，当某些标签仅为15,000美元时，2,500美元并不是微不足道的数额。

让我们看看模型在测试集上的表现如何：

```r 
c(loss, mae) %<-% (model1 %>% evaluate(test_df %>% select(-label), test_df$label, verbose = 0))
paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))
# [1] "Mean absolute error on test set: $2903.54"

c(loss2, mae2) %<-% (model2 %>% evaluate(test_df %>% select(-label), test_df$label, verbose = 0))
paste0("Mean absolute error on test set: $", sprintf("%.2f", mae2 * 1000))
# [1] "Mean absolute error on test set: $3034.21"
```

#### 使用模型进行预测

最后，使用测试集中的数据预测一些房价：

```r 
model<-model2
test_predictions <- model2 %>% predict(test_df %>% select(-label))
test_predictions[ , 1]
#  [1]  7.355123 17.675547 19.973572 32.076920 23.902725 19.619333 26.324997
#  [8] 21.288185 18.688896 21.509230 17.615231 16.178177 15.070681 39.809803
# [15] 20.088022 19.298931 25.012741 21.137566 17.962463 34.908684 10.398927
# [22] 14.056004 19.420004 13.601798 19.115681 24.314400 29.992420 29.077190
# [29]  9.517317 20.587708 18.875362 14.579519 31.711769 23.667265 17.235331
# [36]  7.131071 14.524129 17.223494 18.043385 24.907063 30.337440 26.690973
# [43] 12.751408 38.409725 28.244289 23.861389 24.783478 15.663984 22.516479
# [50] 20.941717 32.775738 19.605915  9.694613 13.944555 33.987289 26.646553
# [57] 11.730441 46.070683 32.869167 23.082375 24.691891 16.289280 15.114546
# [64] 18.146664 21.945438 21.380447 13.035687 21.360455 12.413694  5.442961
# [71] 34.066624 29.388529 24.445929 12.748474 23.918005 18.901218 19.777264
# [78] 21.921507 33.413986  9.202299 19.752489 36.857082 15.715253 12.539826
# [85] 16.558025 18.447201 20.974888 18.670275 21.062746 29.102020 19.141819
# [92] 18.646065 24.691332 41.360500 32.999287 18.146885 34.936665 51.217838
# [99] 25.158466 45.784126 31.594339 19.733921
#
```

#### 结论

本笔记本介绍了一些处理回归问题的技术。

- 均方误差（MSE）是用于回归问题（不同于分类问题）的常见损失函数。

- 同样，用于回归的评估指标也不同于分类。 常见的回归指标是平均绝对误差（MAE）。

- 当输入数据要素的值具有不同范围时，每个要素都应独立缩放。

- 如果训练数据不多，则最好选择一个隐藏层很少的小型网络，以免过度拟合。

- 提前停止是防止过度拟合的有用技术。

[波士顿房价]: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

## 文字分类

*注意：本教程要求TensorFlow版本> = 2.1*

本教程使用评论文本将电影评论分为正面评论或负面评论。这是二进制（或两类）分类的示例，它是一种重要且广泛适用的机器学习问题。

我们将使用[IMDB]数据集，其中包含来自[Internet]电影数据库的50,000个电影评论的文本。这些内容分为25,000条用于培训的评论和25,000条用于测试的评论。训练集和测试集是平衡的，这意味着它们包含相同数量的正面和负面评论。

先启动并加载Keras以及其他一些必需的库。

```r 
library(keras)
library(dplyr)
library(ggplot2)
library(purrr)
```
### 下载电影评论数据集

我们将使用由Bo Pang和Lillian Lee创建的电影评论数据集。在作者的允许下，NLTK重新发布了该数据集。

数据集可在[此处](https://www.kaggle.com/nltkdata/movie-review#movie_review.csv)找到 ，并可从Kaggle UI或使用[pins](https://github.com/rstudio/pins)包下载。

如果要使用[pins](https://github.com/rstudio/pins) ，请按照这里的[教程](https://rstudio.github.io/pins/articles/boards-kaggle.html)注册Kaggle画板。然后，您可以运行：

```r 
library(pins)
board_register("kaggle", token = "/home/wangxh/Soft/kaggle.json")
paths <- pins::pin_get("nltkdata/movie-review", "kaggle")
# 我们只需要 movie_review.csv 文件
path <- paths[1]
```
现在，使用包中的*read_csv*函数将其读取到R中readr。
```r 
df <- readr::read_csv(path)
head(df)
## # A tibble: 6 x 6
##   fold_id cv_tag html_id sent_id text                                 tag  
##     <dbl> <chr>    <dbl>   <dbl> <chr>                                <chr>
## 1       0 cv000    29590       0 films adapted from comic books have… pos  
## 2       0 cv000    29590       1 for starters , it was created by al… pos  
## 3       0 cv000    29590       2 to say moore and campbell thoroughl… pos  
## 4       0 cv000    29590       3 "the book ( or \" graphic novel , \… pos  
## 5       0 cv000    29590       4 in other words , don't dismiss this… pos  
## 6       0 cv000    29590       5 if you can get past the whole comic… pos
```
### 检视数据

让我们花一点时间来理解数据的格式。数据集有6万行，每行代表电影评论。该text列具有实际评论，并且tag 代表向我们显示了该评论的分类情绪。数据集里大约一半的评论是负面的(neg)，另一半是正面的(pos)。

```r 
df$text[1]
## [1] "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before ."

df %>% count(tag)
# # A tibble: 2 x 2
#   tag       n
#   <chr> <int>
# 1 neg   31783
# 2 pos   32937

str(df)
# tibble [64,720 × 6] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
 # $ fold_id: num [1:64720] 0 0 0 0 0 0 0 0 0 0 ...
 # $ cv_tag : chr [1:64720] "cv000" "cv000" "cv000" "cv000" ...
 # $ html_id: num [1:64720] 29590 29590 29590 29590 29590 ...
 # $ sent_id: num [1:64720] 0 1 2 3 4 5 6 7 8 9 ...
 # $ text   : chr [1:64720] "films adapted from comic books have had plenty of # success , whether they're about superheroes ( batman , superm"| __truncated__ # "for starters , it was created by alan moore ( and eddie campbell ) , who # brought the medium to a whole new leve"| __truncated__ "to say moore and # campbell thoroughly researched the subject of jack the ripper would be like # saying michael jac"| __truncated__ "the book ( or \" graphic novel , \" if you # will ) is over 500 pages long and includes nearly 30 more that consi"| # __truncated__ ...
 # $ tag    : chr [1:64720] "pos" "pos" "pos" "pos" ...
 # - attr(*, "spec")=
 #  .. cols(
 #  ..   fold_id = col_double(),
 #  ..   cv_tag = col_character(),
 #  ..   html_id = col_double(),
 #  ..   sent_id = col_double(),
 #  ..   text = col_character(),
 #  ..   tag = col_character()
 #  .. )

```

让我们将数据集分为训练集和测试集两部分：

```r 
training_id <- sample.int(nrow(df), size = nrow(df)*0.8)
training <- df[training_id,]
testing <- df[-training_id,]
```
了解每个评论中单词数量的大致分布情况也很有用。
```r 
df$text %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##    1.00   14.00   21.00   23.06   30.00  179.00
```

### 准备数据

评论（文本）在输入到神经网络之前必须先转换为张量(Tensor)。 首先，我们创建了一个字典和一个整数代表每10,000个最常用的词。在这种情况下，每个评论都将由整数序列表示。

然后，我们可以通过两种方式表示评论：

- 第一种是对数组进行一次热编码，以将其转换为由0和1组成的向量。例如，序列[3，5]将变换为一个10,000维向量，除了索引3和5都是1之外，它们全为零。然后，将其设置为我们网络中的第一层，密集层(dense layer )，即可以处理浮点矢量数据的一层。但是，此方法需要占用大量内存，而且需要使用*num_words * num_reviews*大小矩阵。

- 第二种是可以填充数组，使它们都具有相同的长度，然后创建维度为num_examples * max_length的张量。我们可以使用能够处理此维度的嵌入层(embedding layer )作为网络中的第一层。

在本教程中，我们将使用第二种方法。现在，让我们定义文本向量化层(Text Vectorization layer)，它将负责获取字符串输入并将其转换为张量(Tensor)。

```r 
num_words <- 10000
max_length <- 50
text_vectorization <- layer_text_vectorization(
  max_tokens = num_words, 
  output_sequence_length = max_length, 
)
```
现在，我们需要*adapt*文本向量化层。adapt层将了解数据集中的去重复词汇，并为每个单词分配一个整数值。

```r 
text_vectorization %>% 
  adapt(df$text)
```

您可以看到文本矢量化层如何转换其输入数据的：

```r 
text_vectorization(matrix(df$text[1], ncol = 1))
## tf.Tensor(
## [[  68 2835   30  359 1662   33   91 1056    5  632  631  321   41 7803
##    709 4865 1767   48 7600 1337  398 5161   48    2    1 1808 1800  148
##     17  140  109   90   69    3  359  408   40   30  503  142    0    0
##      0    0    0    0    0    0    0    0]], shape=(1, 50), dtype=int64)
```
### 建立模型

神经网络是通过堆叠层创建的-这需要两个主要的体系结构决策：

- 在模型中使用多少层？

- 每层使用多少个隐藏单元？

在此示例中，输入数据由单词索引数组组成。要预测的标签为0或1(Neg或者Pos)。让我们为这个问题建立一个模型：

```r 
input <- layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)

```
依次堆叠各层以构建分类器：

- 第一层是嵌入层(embedding layer)。该层输入整数编码的词汇表，并为每个单词索引查找嵌入向量。这些向量将用于训练模型。嵌入向量将添加到输出数组，输出结果的维度为：（batch, sequence, embedding）。

- 接下来，global_average_pooling_1d层 层通过对序列维度进行平均，为每个示例返回一个固定长度的输出向量。这允许模型以最简单的方式处理可变长度的输入。

- 该固定长度的输出向量通过管道传输到设置包含有16个隐藏单元的完全连接层(dense layer)。

- 最后一层是密集连接的单个输出节点。使用*sigmoid*激活函数，此值是0到1之间的浮点数，表示概率或置信度。

#### 隐藏的单元

上面的模型在输入和输出之间有两个中间层或“隐藏”层( intermediate or “hidden” layers)。输出的数量(单元、节点或神经元)是该层的表征的空间的维数。换句话说，网络模型再学习一个内部表征的自由度是任意的。

#### 损失函数和优化器

一个模型需要一个损失函数和一个训练优化器。由于本案例是一个二分类问题，模型输出一个概率(带有sigmoid激活的单个单元层)，我们将使用*binary_crossentropy*损失函数(Loss function)。

这不是损失函数的唯一选择，例如，您可以选择*mean_squared_error*。但是，一般来说，*binary_crossenpy*更适合处理概率——它测量概率分布之间的“距离”，或者在我们的例子中，即是真实分布和预测之间的“距离”。

稍后，当我们探讨回归问题(比如，预测房屋价格)时，我们将看到如何使用另一个称为均方误差( mean squared error)的损失函数。

现在，配置模型中使用的优化器和损失函数：

```r 
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)
```

### 训练模型

模型训练使用包含512个样本的小批量数据集进行20个epochs，也就是对x_train和y_train张量中的所有样本进行20次迭代。在训练时，在验证集的10,000个样本上监控模型的损失和准确性:

```r 
history <- model %>% fit(
  training$text,
  as.numeric(training$tag == "pos"),
  epochs = 10,
  batch_size = 512,
  validation_split = 0.2,
  verbose=2
)
## Epoch 1/10
## 81/81 - 1s - loss: 0.6922 - accuracy: 0.5284 - val_loss: 0.6900 - val_accuracy: 0.5717
## Epoch 2/10
## 81/81 - 1s - loss: 0.6872 - accuracy: 0.5616 - val_loss: 0.6823 - val_accuracy: 0.5972
## Epoch 3/10
## 81/81 - 1s - loss: 0.6750 - accuracy: 0.6003 - val_loss: 0.6676 - val_accuracy: 0.6338
## Epoch 4/10
## 81/81 - 1s - loss: 0.6529 - accuracy: 0.6426 - val_loss: 0.6463 - val_accuracy: 0.6536
## Epoch 5/10
## 81/81 - 1s - loss: 0.6250 - accuracy: 0.6713 - val_loss: 0.6251 - val_accuracy: 0.6642
## Epoch 6/10
## 81/81 - 1s - loss: 0.5980 - accuracy: 0.6940 - val_loss: 0.6092 - val_accuracy: 0.6731
## Epoch 7/10
## 81/81 - 1s - loss: 0.5746 - accuracy: 0.7105 - val_loss: 0.5998 - val_accuracy: 0.6771
## Epoch 8/10
## 81/81 - 1s - loss: 0.5557 - accuracy: 0.7259 - val_loss: 0.5940 - val_accuracy: 0.6797
## Epoch 9/10
## 81/81 - 1s - loss: 0.5401 - accuracy: 0.7372 - val_loss: 0.5918 - val_accuracy: 0.6812
## Epoch 10/10
## 81/81 - 1s - loss: 0.5255 - accuracy: 0.7488 - val_loss: 0.5917 - val_accuracy: 0.6831

```

### 评估模型

让我们看看这个模型是如何运行的。将返回两个值。损失值(一个表示我们的误差的数字，越低的值越好)和准确性。

```r 
results <- model %>% evaluate(testing$text, as.numeric(testing$tag == "pos"), verbose = 0)
results
##      loss  accuracy
## 0.5864145 0.6879635
```
这种相当朴素的方法可以达到约68％的精度。使用更高级的方法，模型应接近85％。

### 创建一个随时间变化的准确性和损失图表

*fit*返回一个*keras_training_history*对象，它的*metrics*包含训练期间记录的丢失和度量值( loss and metrics values)。你可以方便地使用它来绘制损失和指标曲线:
```r 
plot(history)
```
```{r fig7, echo=FALSE,out.width="60%", fig.cap ='训练模型的收敛过程和时间2',fig.align='center'} 

knitr::include_graphics('https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_text_classification_files/figure-html/unnamed-chunk-16-1.png')
```
损失和指标的演变，也可以在RStudio浏览器窗格训练中看出。

请注意，训练损失随每个epoch 而减少，训练准确度随每个epoch 而增加。 使用梯度下降优化(gradient descent optimization)时，这是可以预期的-它应在每次迭代中将所需的数量最小化。

[IMDB]:https://keras.rstudio.com/reference/dataset_imdb.html
[Internet]:https://www.imdb.com/

## 使用tfhub中的学习模型

本教程使用评论文本将电影评论分为正面评论或负面评论。这是二进制（或两类）分类的示例，它是一种重要且广泛适用的机器学习问题。

我们将使用[IMDB]数据集，其中包含来自[Internet]电影数据库的50,000个电影评论的文本。这些内容分为25,000条用于培训的评论和25,000条用于测试的评论。训练集和测试集是平衡的，这意味着它们包含相同数量的正面和负面评论。

我们将使用[Keras]构建和培训模型，使用[tfhub]进行迁移学习。我们还将使用tfds来加载IMDB数据集。

先启动并加载Keras以及其他一些必需的库。

```r 
library(keras)
library(tfhub)
library(tfds)
library(tfdatasets)
```
### 下载IMDB数据集

IMDB数据集可在[IMDB reviews]或[tfd]上获得。Keras打包的文件已经经过了预处理，因此对本教程没有用处。

以下代码下载IMDB数据集到您的机器:
```r 
imdb <- tfds_load(
  "imdb_reviews:1.0.0", 
  split = list("train[:60%]", "train[-40%:]", "test"), 
  as_supervised = TRUE
)
summary(imdb)
## This is a dataset for binary sentiment classifica
## ❯ Name: imdb_reviews
## ❯ Version: 1.0.0
## ❯ URLs: http://ai.stanford.edu/~amaas/data/sentiment/
## ❯ Size:
## ❯ Splits:
##  — test ( examples)
##  — train ( examples)
##  — unsupervised ( examples)
## ❯ Schema:
```
*tfds_load*返回一个TensorFlow数据集，是表示元素序列的抽象，其中每个元素由一个或多个组件组成。

要访问数据集的单个元素，您可以使用:
```r 
first <- imdb[[1]] %>% 
  dataset_batch(1) %>% # Used to get only the first example
  reticulate::as_iterator() %>% 
  reticulate::iter_next()
str(first)
## List of 2
##  $ :tf.Tensor([b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it."], shape=(1,), dtype=string)
##  $ :tf.Tensor([0], shape=(1,), dtype=int64)
```

接下来，我们将看到Keras知道如何自动从TensorFlow数据集提取元素，这比在传递给Keras之前将整个数据集加载到RAM中更有效地提高了内存效率。

### 构建模型

神经网络是通过堆叠层来创建的——这需要三个主要的架构决策:

- 如何代表文字？

- 在模型中使用多少层？

- 每层使用多少个隐藏单元？

在本例中，输入数据由句子组成。要预测的标签不是0就是1。

表示文本的一种方法是将句子转换为嵌入向量。 我们可以使用预训练的文本嵌入作为第一层，这将具有三个优点：

* 我们不必担心文本预处理，

* 我们可以从迁移学习中受益，

* 嵌入的大小是固定的，因此处理起来更简单。

在此示例中，我们将使用来自[TensorFlow Hub]的预训练文本嵌入模型[google/tf2-preview/gnews-swivel-20dim/1]。

为了本教程的目的，还需要测试其他三个预先训练过的模型:

- [google/tf2-preview/gnews-swivel-20dim-with-oov/1]是与google/tf2-preview/gnews-swivel-20dim/1相同，但有2.5％的词汇量转换为OOV存储桶。 如果任务的词汇表和模型的词汇表没有完全重叠，则可以提供帮助。

- [google/tf2-preview/nnlm-en-dim50/1]是更大的模型，词汇量约为一百万(1M)，维度为50。

- [google/tf2-preview/nnlm-en-dim128/1]是更加大的词汇模型，词汇量约为1M，维度为128。

让我们首先创建一个Keras层(Keras layer)，它使用TensorFlow Hub中的模型来嵌入句子，并在几个输入数据的示例中了解它。注意，无论输入文本的长度是多少，嵌入层的输出形状都是:(num_examples, embeddding_dimension)。

注意：如果大陆网络无法使用https访问TensorFlow Hub中的模型，我们可以先尝试通过浏览器将模型数据下载到本地(如[tf2-preview_gnews-swivel-20dim_1.tar.gz])，然后解压压缩包到指定路径(如：/tmp/tensorflow_hub/tf2-preview_gnews-swivel-20dim_1)，在调用模型时，便可以直接调用本地的模型数据了()。

```r 
embedding_layer <- layer_hub(handle = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")
embedding_layer(first[[1]])

# 通过调用下载到本地的模型数据
# embedding_layer <- layer_hub(handle = "/tmp/tensorflow_hub/tf2-preview_gnews-swivel-20dim_1")
# embedding_layer(first[[1]])

## tf.Tensor(
## [[ 1.765786   -3.882232    3.9134233  -1.5557289  -3.3362343  -1.7357955
##   -1.9954445   1.2989551   5.081598   -1.1041286  -2.0503852  -0.72675157
##   -0.65675956  0.24436149 -3.7208383   2.0954835   2.2969332  -2.0689783
##   -2.9489717  -1.1315987 ]], shape=(1, 20), dtype=float32)
```

现在让我们构建完整的模型:
```r 
model <- keras_model_sequential() %>% 
  layer_hub(
    handle = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
    input_shape = list(),
    dtype = tf$string,
    trainable = TRUE
  ) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)
## Model: "sequential"
## ________________________________________________________________________________
## Layer (type)                        Output Shape                    Param #
## ================================================================================
## keras_layer_1 (KerasLayer)          (None, 20)                      400020
## ________________________________________________________________________________
## dense_3 (Dense)                     (None, 16)                      336
## ________________________________________________________________________________
## dense_2 (Dense)                     (None, 1)                       17
## ================================================================================
## Total params: 400,373
## Trainable params: 400,373
## Non-trainable params: 0
## ________________________________________________________________________________

```
依次堆叠各层以构建分类器：

1. 第一层是TensorFlow Hub层(TensorFlow Hub layer)。该层使用预先训练好的模型将句子映射到其嵌入向量中。即这里使用的经过预训练的文本嵌入模型([google/tf2-preview/gnews-swivel-20dim/1])将句子拆分为标记，嵌入每个标记，然后组合嵌入层。 结果维度为：（num_examples，embedding_dimension）。

2. 该固定长度的输出矢量通过具有16个隐藏单元的完全连接（密集）层进行传递。

3. 最后一层与单个输出节点紧密连接。 使用*sigmoid*激活函数，该值是0到1之间的浮点数，表示概率或置信度。

### 编译模型

#### 损失函数和优化器

一个模型需要一个损失函数和一个训练优化器。由于本案例是一个二分类问题，模型输出一个概率(带有sigmoid激活的单个单元层)，我们将使用*binary_crossentropy*损失函数(Loss function)。

这不是损失函数的唯一选择，例如，您可以选择*mean_squared_error*。但是，一般来说，*binary_crossenpy*更适合处理概率——它测量概率分布之间的“距离”，或者在我们的例子中，即是真实分布和预测之间的“距离”。

稍后，当我们探讨回归问题(比如，预测房屋价格)时，我们将看到如何使用另一个称为均方误差(mean squared error)的损失函数。

现在，配置模型中使用的优化器和损失函数：

```r 
model %>% 
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )
```

### 模型训练

模型训练使用包含512个样本的小批量数据集进行20个epochs，也就是对x_train和y_train张量中的所有样本进行20次迭代。在训练时，在验证集的10,000个样本上监控模型的损失和准确性:

```r 
history<-model %>% fit(
    imdb[[1]] %>% dataset_shuffle(10000) %>% dataset_batch(512),
    epochs = 20,
    validation_data = imdb[[2]] %>% dataset_batch(512),
    verbose = 2
  )

```

### 评估模型

让我们看看这个模型是如何运行的。将返回两个值。损失值(一个表示我们的误差的数字，越低的值越好)和准确性。

```r 
results <- model %>% 
  evaluate(imdb[[3]] %>%  dataset_batch(512), verbose = 0)
results
##      loss  accuracy
## 0.3169311 0.8660800
```
这种简单的方法可以达到约87％的精度。使用更高级的方法，模型准确度应接近95％。

### 准确性和损失函数的图表

*fit*返回一个*keras_training_history*对象，它的*metrics*包含训练期间记录的丢失和度量值( loss and metrics values)。你可以方便地使用它来绘制损失和指标曲线:
```r 
plot(history)
```
```{r fig8, echo=FALSE,out.width="60%", fig.cap ='训练模型的收敛过程和时间3',fig.align='center'} 

knitr::include_graphics('https://s3.ax1x.com/2020/12/26/rhVyPP.png')
```
[Keras]:https://github.com/rstudio/keras
[tfhub]:https://github.com/rstudio/tfhub
[IMDB reviews]:https://github.com/tensorflow/datasets/blob/master/docs/datasets.md#imdb_reviews
[tfd]:https://github.com/rstudio/tfds
[TensorFlow Hub]:https://github.com/rstudio/tfhub
[google/tf2-preview/gnews-swivel-20dim/1]:https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1

[google/tf2-preview/gnews-swivel-20dim-with-oov/1]:https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1
[google/tf2-preview/nnlm-en-dim50/1]:https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1
[google/tf2-preview/nnlm-en-dim128/1]:https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1
[tf2-preview_gnews-swivel-20dim_1.tar.gz]:https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1?tf-hub-format=compressed


## 过拟合和欠拟合

在之前的两个教程中——对[电影评论进行分类]和[预测房价]——我们看到，我们的模型在验证数据上的准确性在经过若干个epoch的训练后将达到峰值，然后开始下降。

换句话说，我们的模型将过度拟合训练数据。 学习如何应对过度拟合非常重要。 尽管通常可以在训练集上达到很高的准确性，但我们真正想要的是开发出能够很好地预测或者概括测试数据（或之前从未见过的数据）的模型。

过度拟合的反面是欠拟合。 当发现测试数据仍有改进空间时，就会发生欠拟合。 发生这种情况的原因有很多：如模型不够强大，模型过于规范化，或者仅仅是没有经过足够长时间的训练。 这意味着网络尚未学习训练数据中的相关模式。

为了防止过度拟合，最好的解决方案是使用更多的训练数据。 经过更多数据训练的模型自然会更好地推广。 当这不再可能时，下一个最佳解决方案是使用正则化之类的技术。 这些正则化之类的技术限制了模型可以存储的信息的数量和类型。 如果网络模型只能存储少量模式，那么优化过程将迫使它专注于最突出的模式，这些模式具有更好的泛化性。

在本教程中，我们将探讨两种常见的正则化技术：权重正则化(weight regularization)和丢包(dropout)，并使用它们来改进我们的IMDB电影评论分类结果。

先启动并加载Keras以及其他一些必需的库。

```r 
library(keras)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)
```

### 下载IMDB数据集

```r 
num_words <- 1000
imdb <- dataset_imdb(num_words = num_words)
#https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test
```
与在前面的笔记中使用嵌入层不同，这里我们将对句子进行多次热编码。该模型将快速地对训练集进行过拟合。它将被用来演示何时发生过拟合，以及如何应对过拟合。

对列表进行多次热编码意味着将它们转换为0和1的向量。 具体而言，这意味着例如将序列[3，5]转换为一个10,000维向量，该向量除索引3和5是1外，其余的全是零。

```r 
multi_hot_sequences <- function(sequences, dimension) {
  multi_hot <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)) {
    multi_hot[i, sequences[[i]]] <- 1
  }
  multi_hot
}

train_data <- multi_hot_sequences(train_data, num_words)
test_data <- multi_hot_sequences(test_data, num_words)
```
让我们看一下其中一个多次热编码点矢量。 由于单词索引是按频率排序，因此可以预期在索引0附近有更多的1值，如我们在该图中所看到的：

```r 
first_text <- data.frame(word = 1:num_words, value = train_data[1, ])
ggplot(first_text, aes(x = word, y = value)) +
  geom_line() +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
```
```{r fig9, echo=FALSE,out.width="60%", fig.cap ='训练模型的收敛过程和时间3',fig.align='center'} 

knitr::include_graphics('https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_overfit_underfit_files/figure-html/unnamed-chunk-4-1.png')
```

### 过拟合示例

防止过度拟合的最简单方法是减小模型的大小，即减小模型中可学习的参数的数量（由层数和每层单元数确定）。 在深度学习中，模型中可学习参数的数量通常称为模型的“容量”。 直观地讲，具有更多参数的模型将具有更多的“记忆能力”，因此将能够轻松学习训练样本与其目标之间的完美的字典式映射，这种映射没有任何泛化能力，在对以前看不见的数据进行预测时，这将毫无用处。

>始终牢记这一点：深度学习模型往往擅长拟合训练数据，但真正的挑战是**泛化**，而不是拟合。

不幸的是，没有神奇的公式来确定模型的正确大小或体系结构（根据层数或每层的参数的正确大小）。 您将不得不尝试使用一系列不同的体系结构。

为了找到合适的模型尺寸，最好从相对较少的图层和参数开始，然后开始增加图层的大小或添加新的图层，直到看到验证损失的收益递减为止。 让我们在电影评论分类网络上尝试一下。

我们将创建一个仅使用密集层的基础模型，和一个比较小型的模型，并对它们进行进行比较。

#### 建立基础模型

```r 
baseline_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

baseline_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(baseline_model)
## Model: "sequential"
## ________________________________________________________________________________
## Layer (type)                        Output Shape                    Param #
## ================================================================================
## dense_2 (Dense)                     (None, 16)                      16016
## ________________________________________________________________________________
## dense_1 (Dense)                     (None, 16)                      272
## ________________________________________________________________________________
## dense (Dense)                       (None, 1)                       17
## ================================================================================
## Total params: 16,305
## Trainable params: 16,305
## Non-trainable params: 0
## ________________________________________________________________________________

baseline_history <- baseline_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)
## Epoch 1/20
## 49/49 - 4s - loss: 0.5986 - accuracy: 0.6920 - val_loss: 0.4629 - val_accuracy: 0.8026
## ...
## Epoch 20/20
## 49/49 - 0s - loss: 0.1784 - accuracy: 0.9338 - val_loss: 0.4087 - val_accuracy: 0.8425

```

#### 创建一个更小的模型

让我们创建一个包含较少隐藏单位的模型，与我们刚刚创建的基础模型进行比较:

```r 
smaller_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 4, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

smaller_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(smaller_model)
## Model: "sequential_1"
## ________________________________________________________________________________
## Layer (type)                        Output Shape                    Param #
## ================================================================================
## dense_5 (Dense)                     (None, 4)                       4004
## ________________________________________________________________________________
## dense_4 (Dense)                     (None, 4)                       20
## ________________________________________________________________________________
## dense_3 (Dense)                     (None, 1)                       5
## ================================================================================
## Total params: 4,029
## Trainable params: 4,029
## Non-trainable params: 0
## ________________________________________________________________________________

```
并使用相同的数据训练模型:

```r 
smaller_history <- smaller_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)
## Epoch 1/20
## 49/49 - 0s - loss: 0.6219 - accuracy: 0.6821 - val_loss: 0.5364 - val_accuracy: 0.7832
## ...
## Epoch 20/20
## 49/49 - 0s - loss: 0.2952 - accuracy: 0.8787 - val_loss: 0.3320 - val_accuracy: 0.8589
```

#### 创建一个更大的模型

接下来，让我们在这个基准上添加一个容量更大的网络，远远超出了问题所能保证的范围:

```r 
bigger_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

bigger_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(bigger_model)
## Model: "sequential_2"
## ________________________________________________________________________________
## Layer (type)                        Output Shape                    Param #
## ================================================================================
## dense_8 (Dense)                     (None, 512)                     512512
## ________________________________________________________________________________
## dense_7 (Dense)                     (None, 512)                     262656
## ________________________________________________________________________________
## dense_6 (Dense)                     (None, 1)                       513
## ================================================================================
## Total params: 775,681
## Trainable params: 775,681
## Non-trainable params: 0
## ________________________________________________________________________________
```
并使用相同的数据训练模型:
```r 
bigger_history <- bigger_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)
## Epoch 1/20
## 49/49 - 1s - loss: 0.4507 - accuracy: 0.7849 - val_loss: 0.3708 - val_accuracy: 0.8410
## ...
## Epoch 20/20
## 49/49 - 1s - loss: 5.0736e-05 - accuracy: 1.0000 - val_loss: 0.8649 - val_accuracy: 0.8527
```

### 绘制培训和验证损失

现在，让我们绘制3种模型的损耗曲线。较小的网络模型开始过拟合的时间比基线模型稍晚，并且一旦开始过拟合，它的性能下降得更慢。请注意，较大的网络模型仅在一个epoch之后就开始过度拟合，而且是严重过度拟合。网络模型具有的容量越多，将能够更快地对训练数据进行建模（导致较低的训练损失），但它越容易过拟合（导致训练和验证损失之间存在较大差异）。

```r 
compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  smaller_train = smaller_history$metrics$loss,
  smaller_val = smaller_history$metrics$val_loss,
  bigger_train = bigger_history$metrics$loss,
  bigger_val = bigger_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)
  
ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")
```
```{r fig10, echo=FALSE,out.width="60%", fig.cap ='b不同复杂程度的网络模型的拟合过程和时间',fig.align='center'} 

knitr::include_graphics('https://s3.ax1x.com/2020/12/26/rhbVuF.png')
```

### 策略

#### 权重正则化

你可能熟悉奥卡姆剃刀原理(Occam’s Razor principle): 对于某件事有两种解释，最有可能正确的解释是“最简单的”这种解释，也就是做出最少假设的那个解释。这也适用于神经网络学习的模型:给定一些训练数据和一个网络架构，有多个权重值集(多个模型)可以解释数据，简单的模型比复杂的模型更不容易过度拟合。

在这个上下文中，“简单模型”是指参数值分布的熵更少的模型(或者是一个具有更少参数的模型，如我们在上面一节中看到的)。因此，减少过度拟合的一种常见方法是通过强制网络的权重仅采用较小的值来对网络的复杂性施加约束，这使得权重值的分布更加“规则”。 这称为“权重调整”，这是通过向网络的损失函数添加与拥有大权重相关的代价来实现的。这种代价有两个方面:

- L1正则化，其中增加的成本与权重系数的绝对值成比例(即与权重的“L1范数”成比例)。
- L2正则化，其中增加的成本与权重系数的值的平方成正比（即，权重的所谓“ L2范数”）。 L2正则化在神经网络中也称为权重衰减。 不要让不同的名字迷惑你:重量衰减在数学上和L2正则化是完全一样的。

在Keras中，权重正则化是通过将权重正则化实例传递给层来添加的。现在让我们将L2权重正则化添加到基线模型中。

```r 
l2_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words,
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 1, activation = "sigmoid")

l2_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

l2_history <- l2_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)
## Epoch 1/20
## 49/49 - 0s - loss: 0.5858 - accuracy: 0.7506 - val_loss: 0.4587 - val_accuracy: 0.8314
## ...
## Epoch 20/20
## 49/49 - 0s - loss: 0.3177 - accuracy: 0.8785 - val_loss: 0.3523 - val_accuracy: 0.8592
```
regularizer_l2(l = 0.001)表示该层权重矩阵中的每一个系数都会使网络的总损耗增加0.001 * weight_coefficient_value。注意，因为这个惩罚只在训练时添加，所以这个网络模型在训练时的损失要比在测试时高得多。

以下是L2正则化惩罚的影响:

```r 
compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  l2_train = l2_history$metrics$loss,
  l2_val = l2_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)
  
ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")
```
```{r fig11, echo=FALSE,out.width="60%", fig.cap ='l2权重正则化处理的网络模型的拟合过程',fig.align='center'} 

knitr::include_graphics('https://s3.ax1x.com/2020/12/26/rhbv26.png')
```
如您所见，L2正则化模型比基础模型更能抵抗过拟合，即使两个模型具有相同数量的参数。

#### 丢包

丢包(dropout)是神经网络最有效和最常用的正则化技术之一，由Hinton和他在多伦多大学的学生开发。丢包(dropout)应用于一个层，由训练过程中随机“Dropout”(即设置为零)该层的一些输出特征组成。假设给定的层通常会在训练过程中为给定的输入样本返回一个向量[0.2,0.5,1.3,0.8,1.1];应用dropout后，这个向量将有几个随机分布的零项，例如[0,0.5,1.3,0,1.1]。“退出率”("dropout rate")是被归零的特征的分数;通常设置在0.2到0.5之间。在测试时，没有单位被删除，相反，该层的输出值被缩小了一个等于删除率的因子，以便平衡更多的单位比训练时活跃的事实。

在Keras中，您可以通过*layer_dropout*在网络模型中引入丢包，该丢包将立即应用于图层的输出。

让我们在IMDB网络中添加两个dropout层，看看它们在减少过拟合方面做得如何:
```r 
dropout_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 1, activation = "sigmoid")

dropout_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

dropout_history <- dropout_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)
```
它的效果如何?添加dropout是对基线模型的明显改进。

```r 
compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  dropout_train = dropout_history$metrics$loss,
  dropout_val = dropout_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)
  
ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")

```
```{r fig12, echo=FALSE,out.width="60%", fig.cap ='丢包正则化处理的网络模型的拟合过程',fig.align='center'} 

knitr::include_graphics('https://s3.ax1x.com/2020/12/26/rhOF3t.png')
```

总结一下，以下是防止神经网络过度拟合的最常见方法:

- 获取更多的训练数据。

- 请降低网络容量。

- 增加权重正则化处理

- 增加丢包正则化处理

本指南中没有涉及的两种重要方法是数据增强(Data augmentation )和批归一化(Batch normalization)。

[电影评论进行分类]:https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_overfit_underfit/tutorial_basic_text_classification.html
[预测房价]:https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_overfit_underfit/tutorial_basic_regression.html


## 保存和恢复模型

可以在训练后和训练中保存模型进度。这意味着一个模型可以在它停止的地方恢复，避免长时间的训练。保存还意味着您可以共享您的模型，其他人可以重新创建您的工作。在发布研究模型和技术时，与大多数机器学习实践者分享:

- 创建模型的代码

- 训练过的权重，或模型的参数

共享这些数据可以帮助其他人理解模型的工作方式，并使用新数据自己尝试。

### 选项

有很多不同的方法来保存TensorFlow模型——这取决于你使用的API。本指南使用Keras，一个高级API来构建和训练TensorFlow模型。对于其他方法，参见[TensorFlow Save and Restore guide]或[Saving in eager]。

### 设置

我们将使用[MNIST]数据集训练我们的模型来演示保存训练后的权重。为了加快这些演示的运行速度，只使用前1000个示例:

```r 
library(keras)

mnist <- dataset_mnist()

c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

train_labels <- train_labels[1:1000]
test_labels <- test_labels[1:1000]

train_images <- train_images[1:1000, , ] %>%
  array_reshape(c(1000, 28 * 28))
train_images <- train_images / 255

test_images <- test_images[1:1000, , ] %>%
  array_reshape(c(1000, 28 * 28))
test_images <- test_images / 255
```
### 定义一个模型

让我们构建一个简单的模型，我们将使用它来演示保存和加载权重。

```r 
# 返回一个短序列模型
create_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 10, activation = "softmax")
  model %>% compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = list("accuracy")
  )
  model
}

model <- create_model()
summary(model)
## Model: "sequential"
## ________________________________________________________________________________
## Layer (type)                        Output Shape                    Param #
## ================================================================================
## dense_1 (Dense)                     (None, 512)                     401920
## ________________________________________________________________________________
## dropout (Dropout)                   (None, 512)                     0
## ________________________________________________________________________________
## dense (Dense)                       (None, 10)                      5130
## ================================================================================
## Total params: 407,050
## Trainable params: 407,050
## Non-trainable params: 0
## ________________________________________________________________________________
## 

```

### 保存整个模型

调用*save_model_\** 将模型的架构、权重和训练配置保存在单个文件/文件夹中。这允许您导出模型，以便在不访问原始代码的情况下使用它。由于优化器状态已经恢复，您可以从中断的位置恢复训练。

保存模型的常用函数及其加载函数：
- save_model_hdf5() 和 load_model_hdf5

- save_model_tf() 和 load_model_tf()

- save_model_weights_hdf5() 和 load_model_weights_hdf5():

- save_model_weights_tf() 和 load_model_weights_tf():

保存一个全功能模型是非常有用的——你可以在TensorFlow.js (HDF5, Saved Model)中加载它们，然后在web浏览器中训练和运行它们，或者使用TensorFlow Lite (HDF5, Saved Nodel)将它们转换到移动设备上运行。

### SAVEDMODEL格式

SavedModel格式是一种序列化模型的方法。以这种格式保存的模型可以使用*load_model_t()*恢复，并且与TensorFlow服务兼容。[SavedModel指南]详细介绍了如何服务/检查SavedModel。下面的部分演示了保存和恢复模型的步骤。下面的部分演示了保存和恢复模型的步骤。

```r 
model <- create_model()

model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

## SavedModel 格式是一个包含protobuf二进制文件和Tensorflow检查点的目录。
model %>% save_model_tf("model")
##检查保存的模型目录:
list.files("model")

## 从保存的模型中重新加载一个新的Keras模型:
new_model <- load_model_tf("model")
summary(new_model)
```

### HDF5格式

Keras使用HDF5标准提供了基本的保存格式。

```r 
model <- create_model()
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

model %>% save_model_hdf5("my_model.h5")

## 现在从文件中重新创建模型:
new_model <- load_model_hdf5("my_model.h5")
summary(new_model)
```
这种技术可以保存所有信息：

- 权重值

- 模型的配置（架构）

- 优化器配置

Keras的SavedModel通过检查架构来保存模型。目前，SavedModel不能保存TensorFlow优化器(在tf$train中)。在使用SavedModel时，您将需要在加载后重新编译模型，并且您将丢失优化器的状态。

### 保存自定义对象

如果您使用SavedModel格式，则可以跳过本节。HDF5和SavedModel的关键区别在于，HDF5使用对象配置来保存模型架构，而SavedModel则保存执行图。

因此，SavedModels能够保存自定义对象，如子类模型和自定义层，而不需要原始代码。

要将自定义对象保存到HDF5，必须执行以下操作:

1. 在对象中定义一个get_config方法，还有一个from_config类方法。

    - get_config()返回一个json可序列化的参数字典，其中包含重新创建对象所需的参数。

    - from_config(config)使用从get_config()返回的配置来创建一个新的对象。默认情况下，该函数将使用config作为初始化参数。

2. 在加载模型时将对象传递给*custom_objects*参数。参数必须是一个将字符串类名映射到类定义的命名列表。例如 load_keras_model_hdf5(path, custom_objects=list("CustomLayer" = CustomLayer)) 

关于custom_objects()和get_config()的示例，请参阅[从头编写层和模型教程]，**貌似这部分内容还没完成**。


### 在训练期间保存检查点

在训练期间和结束时自动保存检查点是很有用的。通过这种方式，你可以使用一个训练过的模型，而不必重新训练它，或者在你离开的地方接上训练，以防训练过程中断。

callback_model_checkpoint是执行此任务的回调函数。

回调函数接受两个参数来配置检查点。默认情况下，save_weights_only设置为false，这意味着保存完整的模型——包括架构和配置。然后，您可以按照前一段所述的方式恢复模型。

现在，让我们专注于保存和恢复权重。在下面的代码片段中，我们将save_weights_only设置为true，因此在恢复时需要模型定义。


#### 使用检查点回调

训练模型并给它传递callback_model_checkpoint:

```r 
checkpoint_path <- "checkpoints/cp.ckpt"

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  verbose = 0
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10, 
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback),  # pass callback to training
  verbose = 2
)
```
检查创建的文件:

```r 
list.files(dirname(checkpoint_path))
```
创建一个新的未经训练的模型。当仅从权重恢复模型时，您必须拥有与原始模型具有相同架构的模型。由于它是相同的模型架构，我们可以共享权重，尽管它是模型的不同实例。

现在重建一个新的，未经训练的模型，并在测试集上评估它。未经训练的模型将在概率水平(~7% 精确度)执行:

```r 
fresh_model <- create_model()
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)
##     loss accuracy
## 2.326613 0.069000

```

然后从最新的检查点(epoch 10)加载权重，并重新评估:

```r 
fresh_model %>% load_model_weights_tf(filepath = checkpoint_path)
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)
##      loss  accuracy
## 0.4079803 0.8700000

```

#### 检查点回调选项

另外，您可以决定仅保存最佳模型，默认情况下，最佳模型定义为验证损失最小。 有关更多信息，请参见[callback_model_checkpoint的文档]。

```r 
checkpoint_path <- "checkpoints/cp.ckpt"

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 1
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10, 
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback), # pass callback to training,
  verbose = 2
)

list.files(dirname(checkpoint_path))
```

#### 这些文件是什么?

上上面的代码将权重存储到[检查点格式] ([checkpoint-formatted])的文件集合中，这些文件仅以二进制格式包含训练过的权重。 检查点包含：

- 一个或多个包含模型权重的碎片。

- 一个索引文件，指示哪些权重存储在哪个分片中。

如果您仅在一台机器上训练模型，则后缀为 *.s-data-00000-of-00001*。

### 手动保存权重

您了解了如何将权重加载到模型中。手动保存它们使用save_model_weights_tf函数也一样简单。

```r 
# 保存权重
model %>% save_model_weights_tf("checkpoints/cp.ckpt")

# 创建一个新的模型实例
new_model <- create_model()

# 恢复权重进入翻译页面
new_model %>% load_model_weights_tf('checkpoints/cp.ckpt')

# 评价模型
new_model %>% evaluate(test_images, test_labels, verbose = 0)
##     loss accuracy
##  0.39937  0.86800

```


[TensorFlow Save and Restore guide]:https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/todo
[Saving in eager]:https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/todo
[SavedModel指南]:https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/todo
[从头编写层和模型教程]:https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/todo
[callback_model_checkpoint的文档]:https://tensorflow.rstudio.com/keras/reference/callback_model_checkpoint.html
[检查点格式]:https://www.tensorflow.org/guide/saved_model#save_and_restore_variables
[checkpoint-formatted]:https://www.tensorflow.org/guide/saved_model#save_and_restore_variables