from FullyConnectedNet import *

class NeuralNetwork:
  def __init__(self, layers, optimizer, loss_fn=CrossEntropyLoss()):
    self.layers = layers
    self.loss_fn = loss_fn
    self.optimizer = optimizer

  def forward(self, X, is_training=True):
    for layer in self.layers:
      if isinstance(layer, BatchNormalization):
        X = layer.forward(X, is_training=is_training)
      else:
        X = layer.forward(X)
    return X

  def backward(self, dZ):
    for layer in reversed(self.layers):
      if isinstance(layer, Softmax):
        dZ = layer.backward(dZ)
      elif isinstance(layer, BatchNormalization):
        dZ = layer.backward(dZ)
      else:
        dZ = layer.backward(dZ, self.optimizer)

  def train_step(self, X, y_true):
    A = self.forward(X, is_training=True)
    loss = self.loss_fn.forward(A, y_true)
    dZ = self.loss_fn.backward()
    self.backward(dZ)
    return loss

  def predict(self, X):
    A = self.forward(X, is_training=False)
    return np.argmax(A, axis=0)

  def save_activations(self):
    activations = []
    for layer in self.layers:
      if isinstance(layer, Tanh):
        activations.append(layer.activations.flatten())
    return activations


## 计算准确率
def accuracy(y_true, y_pred):
  return np.mean(y_true == y_pred)

## 训练函数
def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, save_interval=10):
  num_samples = X_train.shape[0]
  train_losses = [] # 损失函数值
  val_losses = []
  activations_history = []

  for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, num_samples, batch_size):
      X_batch = X_train[i:i+batch_size].T
      y_batch = y_train[i:i+batch_size].T
      loss = model.train_step(X_batch, y_batch)
      epoch_loss += loss
    
    train_losses.append(epoch_loss / (num_samples // batch_size))

    # 计算验证集损失
    val_loss = 0
    for i in range(0, X_val.shape[0], batch_size):
      X_val_batch = X_val[i:i+batch_size].T
      y_val_batch = y_val[i:i+batch_size].T
      A_val = model.forward(X_val_batch, is_training=False)
      val_loss += model.loss_fn.forward(A_val, y_val_batch)
  
    val_losses.append(val_loss / (X_val.shape[0] // batch_size))

    if (epoch%10 == 0): print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # 定期保存权值
    if (epoch + 1) % save_interval == 0:
      activations_history.append(model.save_activations())
      
  return train_losses, val_losses, activations_history   


def plot_activations_distribution(activations_history, save_interval, save_dir='activations_distribution.png'):
  num_layers = len(activations_history[0])
  num_intervals = len(activations_history)

  fig, axes = plt.subplots(num_layers, num_intervals, figsize=(80, 10))
  if num_layers == 1:
    axes = [axes]
  if num_intervals == 1:
    axes = [[ax] for ax in axes]

  for layer_idx in range(num_layers):
    for interval_idx in range(num_intervals):
      activations = activations_history[interval_idx][layer_idx]
      mean = np.mean(activations)
      std = np.std(activations)
      axes[layer_idx][interval_idx].hist(activations, bins=50)
      axes[layer_idx][interval_idx].set_title(f"Layer {layer_idx+1}, Interval {interval_idx+1}")
      axes[layer_idx][interval_idx].set_xlabel("Activations")
      axes[layer_idx][interval_idx].set_ylabel("Frequency")
      
      # 添加均值和方差的标注
      axes[layer_idx][interval_idx].annotate(
        f"Mean: {mean:.2f}\nStd: {std:.2f}",
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        ha='right',
        va='top',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
      )

  plt.savefig(save_dir)
  plt.show()