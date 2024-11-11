import numpy as np

class Optimizer:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def update(self, layer):
    pass

class GD(Optimizer):
  def update(self, layer):
    layer.W -= self.learning_rate * layer.dW
    layer.b -= self.learning_rate * layer.db

class Momentum(Optimizer):
  def __init__(self, learning_rate, momentum=0.9):
    super().__init__(learning_rate)
    self.momentum = momentum
    self.vW = {}
    self.vb = {}

  def update(self, layer):
    if layer not in self.vW:
      self.vW[layer] = np.zeros_like(layer.W)
      self.vb[layer] = np.zeros_like(layer.b)
    self.vW[layer] = self.momentum * self.vW[layer] + self.learning_rate * layer.dW
    self.vb[layer] = self.momentum * self.vb[layer] + self.learning_rate * layer.db
    layer.W -= self.vW[layer]
    layer.b -= self.vb[layer]

class Adam(Optimizer):
  def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    super().__init__(learning_rate)
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.mW = {}
    self.mb = {}
    self.vW = {}
    self.vb = {}
    self.t = 0

  def update(self, layer):
    if layer not in self.mW:
      self.mW[layer] = np.zeros_like(layer.W)
      self.mb[layer] = np.zeros_like(layer.b)
      self.vW[layer] = np.zeros_like(layer.W)
      self.vb[layer] = np.zeros_like(layer.b)
    self.t += 1
    self.mW[layer] = self.beta1 * self.mW[layer] + (1 - self.beta1) * layer.dW
    self.mb[layer] = self.beta1 * self.mb[layer] + (1 - self.beta1) * layer.db
    self.vW[layer] = self.beta2 * self.vW[layer] + (1 - self.beta2) * (layer.dW ** 2)
    self.vb[layer] = self.beta2 * self.vb[layer] + (1 - self.beta2) * (layer.db ** 2)
    mWh = self.mW[layer] / (1 - self.beta1 ** self.t)
    mbh = self.mb[layer] / (1 - self.beta1 ** self.t)
    vWh = self.vW[layer] / (1 - self.beta2 ** self.t)
    vbh = self.vb[layer] / (1 - self.beta2 ** self.t)
    layer.W -= self.learning_rate * mWh / (np.sqrt(vWh) + self.epsilon)
    layer.b -= self.learning_rate * mbh / (np.sqrt(vbh) + self.epsilon)