class BatchNormalization:
  def __init__(self, momentum=0.9, epsilon=1e-5):
    self.momentum = momentum
    self.epsilon = epsilon
    self.running_mean = None
    self.running_var = None
    self.gamma = None
    self.beta = None

  def forward(self, X, is_training=True):
    if self.gamma is None:
      self.gamma = np.ones(X.shape[0])
      self.beta = np.zeros(X.shape[0])

    if is_training:
      mean = np.mean(X, axis=1, keepdims=True)
      var = np.var(X, axis=1, keepdims=True)

      if self.running_mean is None:
        self.running_mean = mean
        self.running_var = var
      else:
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

      X_norm = (X - mean) / np.sqrt(var + self.epsilon)
      out = self.gamma.reshape(-1, 1) * X_norm + self.beta.reshape(-1, 1)

      self.X_norm = X_norm
      self.X = X
      self.mean = mean
      self.var = var
    else:
      X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
      out = self.gamma.reshape(-1, 1) * X_norm + self.beta.reshape(-1, 1)

    return out

  def backward(self, dout):
    m = dout.shape[1]
    dX_norm = dout * self.gamma.reshape(-1, 1)
    dvar = np.sum(dX_norm * (self.X - self.mean) * -0.5 * (self.var + self.epsilon) ** -1.5, axis=1, keepdims=True)
    dmean = np.sum(dX_norm * -1 / np.sqrt(self.var + self.epsilon), axis=1, keepdims=True) + dvar * np.mean(-2 * (self.X - self.mean), axis=1, keepdims=True)
    dX = dX_norm / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.X - self.mean) / m + dmean / m

    self.dgamma = np.sum(dout * self.X_norm, axis=1)
    self.dbeta = np.sum(dout, axis=1)

    return dX