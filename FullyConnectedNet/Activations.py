import numpy as np

class Softmax:
  def forward(self, Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) #防止指数爆炸
    self.A = expZ / np.sum(expZ, axis=0, keepdims=True)
    return self.A

  def backward(self, dA):
    return dA

class ReLU:
  def forward(self,Z):
    self.Z = Z
    self.A = np.maximum(0, Z)
    self.activations = self.A
    return self.A

  def backward(self, dA, learning_rate=None):
    dZ = np.array(dA, copy=True)
    dZ[self.Z <= 0] = 0
    return dZ

class Sigmoid:
  def forward(self, Z):
    self.A = 1 / (1 + np.exp(-Z))
    self.activations = self.A
    return self.A

  def backward(self, dA, learning_rate=None):
    return dA * self.A * (1 - self.A)

class Tanh:
  def forward(self, Z):
    self.A = np.tanh(Z)
    self.activations = self.A
    return self.A
  def backward(self, dA, learning_rate=None):
    return dA * (1 - np.square(self.A))

class LeakyReLU:
  def __init__(self, alpha=0.01):
    self.alpha = alpha

  def forward(self, Z):
    self.Z = Z
    self.A = np.where(Z > 0, Z, Z * self.alpha)
    self.activations = self.A
    return self.A

  def backward(self, dA, learning_rate=None):
    dZ = np.array(dA, copy=True)
    dZ[self.Z <= 0] *= self.alpha
    return dZ