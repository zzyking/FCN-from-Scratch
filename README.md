# FCN-from-Scratch
ML Assignment 1
- 任务：训练一个**全连接神经网络**分类器，完成图像分类 (20 points)
- 数据：[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

本实验使用 `numpy` 构建了一个三层全连接神经网络，采用了模块化设计，在 CIFAR-10 数据集进行训练和测试，实现了超过50%的分类准确率

网络架构如下：
```plaintext
           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   

               Input   #####      3   32   32
             Flatten   ||||| -------------------         0
                       #####        3072
(BatchNormalization)    μ|σ  -------------------         0
                       #####        3072                 
               Dense   XXXXX -------------------    393216
          Activation   #####         128
(BatchNormalization)    μ|σ  -------------------
                       #####         128
               Dense   XXXXX -------------------      8192
          Activation   #####          64
(BatchNormalization)    μ|σ  -------------------
                       #####          64
               Dense   XXXXX -------------------       640
             softmax   #####          10
```
