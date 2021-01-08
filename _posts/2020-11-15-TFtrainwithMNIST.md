---
layout: post
title: TensorFlow 在 MNIST 中的应用（一）
---

### 是大家



```r
library(tensorflow)
library(keras)
library(dplyr)
library(ggplot2)
library(reshape2)
library(tfestimators)

# 初始化数据目录
data_dir<-"mnist-data"
dir.create(data_dir,recursive=TRUE,showWarnings=FALSE)

# 下载MNIST数据集，读入R
sources <- list(
  train = list(
    x = "/home/wangxh/Work/tftest/Test1/Tensorflowbook.ch9/train-images-idx3-ubyte.gz",
    y = "/home/wangxh/Work/tftest/Test1/Tensorflowbook.ch9/train-labels-idx1-ubyte.gz" ),
  
  test = list(
    x = "/home/wangxh/Work/tftest/Test1/Tensorflowbook.ch9/t10k-images-idx3-ubyte.gz",
    y = "/home/wangxh/Work/tftest/Test1/Tensorflowbook.ch9/t10k-labels-idx1-ubyte.gz" )
)

# 读取MNIST文件(该文件是以IDX格式编码)
read_idx <- function(file) {
  
  # 创建读取文件的二进制连接
  conn <- gzfile(file, open = "rb")
  #用来注册执行exit()函数前执行的终止处理程序。
  on.exit(close(conn), add = TRUE)
  
  # 以4个字节的序列形式读入‘幻数’
  magic <- readBin(conn, what = "raw", n = 4, endian = "big")
  ndims <- as.integer(magic[[4]])
  
  # 读取维度(32位的整数)
  dims <- readBin(conn, what = "integer", n = ndims, endian = "big")
  
  # 其余部分作为原始向量读入
  data <- readBin(conn, what = "raw", n = prod(dims), endian = "big")
  
  # 转换为一个整数向量
  converted <- as.integer(data)
  
  # 返回1维的数组的
  if (length(dims) == 1)
    return(converted)
  
  # 将3D数据打包到矩阵中
  matrix(converted, nrow = dims[1], ncol = prod(dims[-1]), byrow = TRUE)
}

mnist <- rapply(sources, classes = "character", how = "list", function(url) {
  
  # 下载URL的idx文件
  target <- file.path(data_dir, basename(url))
  if (!file.exists(target))
    download.file(url, target)
  
  # 读取idx格式数据
  read_idx(target)
  
})


# 转换训练集和测试集数值归一化为0-1范围，Mnist采用的像素范围最大值是255
max(mnist$train$y)
# 255
max(mnist$test$x)
# 255
mnist$train$x <- mnist$train$x / max(mnist$train$x)
mnist$test$x <- mnist$test$x / max(mnist$test$x)

# 尝试为随机的36个图像的样本绘制图，显示像素强度
n <- 36
indices <- sample(nrow(mnist$train$x), size = n)
data <- array(mnist$train$x[indices, ], dim = c(n, 28, 28))
melted <- melt(data, varnames = c("image", "x", "y"), value.name = "intensity")
ggplot(melted, aes(x = x, y = y, fill = intensity)) +
  geom_tile() +
  scale_fill_continuous(name = "Pixel Intensity") +
  scale_y_reverse() +
  facet_wrap(~ image, nrow = sqrt(n), ncol = sqrt(n)) +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(),
    panel.spacing = unit(0, "lines"),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +
  labs(
    title = "MNIST Image Data",
    subtitle = "Visualization of a sample of images contained in MNIST data set.",
    x = NULL,
    y = NULL
  )


# 构造线性分类器，这部分再tensoflow1.8上可以执行，但是再2.4上面会提示AttributeError: module 'tensorflow.python.feature_column.feature_column' has no attribute 'numeric_column'错误
classifier <- linear_classifier(
  feature_columns = feature_columns(
    column_numeric("x", shape = shape(784L))
  ),
  n_classes = 10L  #10位数字
)

# 构造输入函数生成器
mnist_input_fn <- function(data, ...) {
  input_fn(
    data,
    response = "y",
    features = "x",
    batch_size = 128,
    ...
  )
}

# 训练分类器
train(classifier, input_fn = mnist_input_fn(mnist$train), steps = 200)

# 在测试数据集上评估分类器
evaluate(classifier, input_fn = mnist_input_fn(mnist$test), steps = 200)
# A tibble: 1 x 4
#   accuracy average_loss  loss global_step
#      <dbl>        <dbl> <dbl>       <dbl>
# 1    0.905        0.345  43.6         200

# 使用我们的分类器来预测测试数据集子集的标签
predictions <- predict(classifier, input_fn = mnist_input_fn(mnist$test))

n <- 20
indices <- sample(nrow(mnist$test$x), n)
classes <- vapply(indices, function(i) {
  predictions$classes[[i]]
}, character(1))

data <- array(mnist$test$x[indices, ], dim = c(n, 28, 28))
melted <- melt(data, varnames = c("image", "x", "y"), value.name = "intensity")
melted$class <- classes

image_labels <- setNames(
  sprintf("Predicted: %s\nActual: %s", classes, mnist$test$y[indices]),
  1:n
)

ggplot(melted, aes(x = x, y = y, fill = intensity)) +
  geom_tile() +
  scale_y_reverse() +
  facet_wrap(~ image, ncol = 5, labeller = labeller(image = image_labels)) +
  theme(
    panel.spacing = unit(0, "lines"),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +
  labs(
    title = "MNIST Image Data",
    subtitle = "Visualization of a sample of images contained in MNIST data set.",
    x = NULL,
    y = NULL
  )


```