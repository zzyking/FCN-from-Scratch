from FullyConnectedNet import *
from argparse import ArgumentParser
import os

if __name__ == "__main__":
  X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py')
  
  X_train = normalize(X_train)
  X_test = normalize(X_test)
  y_train = one_hot_encode(y_train, 10)
  y_test = one_hot_encode(y_test, 10)

  parser = ArgumentParser()
  parser.add_argument('--init_method', type=str)
  parser.add_argument('--weight_scale', type=float)
  parser.add_argument('--learning_rate', type=float)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--fig_dir', type=str)
  parser.add_argument('--activations_dir', type=str)
  
  args = parser.parse_args()

  # 定义网络结构
  layers = [
    FullyConnectedLayer(3072, 128, weight_scale=args.weight_scale, init_method=args.init_method),
    Tanh(),
    FullyConnectedLayer(128, 64, weight_scale=args.weight_scale, init_method=args.init_method),
    Tanh(),
    FullyConnectedLayer(64, 10, weight_scale=args.weight_scale, init_method=args.init_method),
    Softmax()
  ]

  # 选择优化器
  optimizer = Adam(learning_rate=args.learning_rate)

  # 选择损失函数
  loss_fn = CrossEntropyLoss()

  model = NeuralNetwork(layers, optimizer, loss_fn)

  # 训练模型
  train_losses, val_losses, activations_history = train(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=args.batch_size, save_interval=20)

  # 计算测试集准确率
  test_preds = model.predict(X_test.T)
  test_acc = accuracy(y_test.argmax(axis=1), test_preds)
  print(f"Test Accuracy: {test_acc:.4f}")  
  

  # 绘制训练损失和验证损失曲线
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.plot(train_losses, label='Train Loss')
  plt.plot(val_losses, label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Train Loss')
  plt.legend()

  if not os.path.exists(args.fig_dir):
    os.makedirs(args.fig_dir)

  plt.savefig(os.path.join(args.fig_dir, 'loss_curves.png'))
  
  if not os.path.exists(args.activations_dir):
    os.makedirs(args.activations_dir)
  # 绘制权值分布
  plot_activations_distribution(activations_history, save_interval=20, save_dir=os.path.join(args.activations_dir, 'activations_distribution.png'))