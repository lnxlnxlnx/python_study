import tensorflow as tf
from keras import layers, models
import os
import pickle
import numpy as np

# 禁用XLA优化器
os.environ['TF_XLA_ENABLE'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# python3的解压代码
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 手动加载本地CIFAR-10数据
def load_cifar10_data(data_dir):
    # 加载训练数据
    x_train = []
    y_train = []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        data_dict = unpickle(file_path)
        x_train.extend(data_dict[b'data'])
        y_train.extend(data_dict[b'labels'])
    x_train = np.array(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)

    # 加载测试数据
    file_path = os.path.join(data_dir, 'test_batch')
    data_dict = unpickle(file_path)
    x_test = np.array(data_dict[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(data_dict[b'labels'])

    return (x_train, y_train), (x_test, y_test)

# 本地数据集目录
data_dir = '/home/lnx/.keras/datasets/cifar-10-python/cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 减少训练数据和测试数据的样本数量
num_train_samples = 100000  # 选择前100000个训练样本
num_test_samples = 20000    # 选择前20000个测试样本
x_train = x_train[:num_train_samples]
y_train = y_train[:num_train_samples]
x_test = x_test[:num_test_samples]
y_test = y_test[:num_test_samples]

# 构建CNN模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# 编译与训练
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))