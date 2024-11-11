import numpy as np

class CrossEntropyLoss:
  def forward(self, A, y_true):
    m = y_true.shape[1]
    self.A = A
    self.y_true = y_true
    y_true_indices = np.argmax(y_true, axis=0)
    log_probs = -np.log(A[y_true_indices, range(m)])
    loss = np.sum(log_probs) / m
    return loss

  def backward(self):
    m = self.y_true.shape[1]
    dZ = self.A - self.y_true
    return dZ / m

class MSELoss:
  def forward(self, A, y_true):
    self.A = A
    self.y_true = y_true
    return np.mean((A - y_true) ** 2)

  def backward(self):
    m = self.y_true.shape[1]
    return 2 * (self.A - self.y_true) / m