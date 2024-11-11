from FullyConnectedNet import *

class FullyConnectedLayer:
  def __init__(self, input_size, output_size, weight_scale=0.1, init_method='normal'):
    self.input_size = input_size
    self.output_size = output_size
    self.init_method = init_method

    ## 权重和偏置初始化
    if init_method == 'normal':
      self.W = np.random.randn(output_size, input_size) * weight_scale
    elif init_method == 'uniform':
      self.W = np.random.uniform(low=-weight_scale, high=weight_scale, size=(output_size, input_size))
    elif init_method == 'equal':
      self.W = np.full((output_size, input_size), weight_scale)
    elif init_method == 'xavier':
      self.W = np.random.randn(output_size, input_size) * np.sqrt(1.0 / (input_size + output_size))
    elif init_method == 'kaiming':
      self.W = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
    else:
      raise ValueError("Unsupported initialization method")

    self.b = np.zeros((output_size, 1))

  def forward(self, X):
    ## 前向传播，计算 Z = W * X + b
    self.X = X
    self.Z = np.dot(self.W, X) + self.b
    return self.Z

  def backward(self, dZ, optimizer):
    ## 反向传播，计算 dW, db, dX 并更新 W, b
    m = self.X.shape[1] # 样本数
    self.dW = np.dot(dZ, self.X.T) / m
    self.db = np.sum(dZ, axis=1, keepdims=True) / m
    dX = np.dot(self.W.T, dZ)

    optimizer.update(self)

    return dX # 返回上一层的梯度
