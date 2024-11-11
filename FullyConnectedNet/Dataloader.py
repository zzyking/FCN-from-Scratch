import numpy as np
import pickle

## 加载数据集
def load_CIFAR_batch(filename):
  with open(filename, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def load_CIFAR10(cifar10_dir):
  xs, ys = [], []
  for i in range(1,6):
    train_batch = load_CIFAR_batch(cifar10_dir + "/data_batch_" + str(i))
    xs.append(train_batch[b'data'])
    ys.append(train_batch[b'labels'])
  X_train = np.concatenate(xs)
  y_train = np.concatenate(ys)

  test_batch = load_CIFAR_batch(cifar10_dir + "/test_batch")
  X_test = np.array(test_batch[b'data'])
  y_test = np.array(test_batch[b'labels'])

  return X_train, y_train, X_test, y_test

## 像素值归一化
def normalize(X): return X / 255.0

## 独热编码
def one_hot_encode(y, num_classes):
  one_hots = np.zeros((y.shape[0], num_classes))
  one_hots[np.arange(y.shape[0]), y] = 1
  return one_hots