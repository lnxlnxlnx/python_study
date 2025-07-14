import os
# 禁用XLA优化器
os.environ['TF_XLA_ENABLE'] = '0'
# 移除未知的标志
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_disable_hlo_passes'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from keras import layers, models, Input
import numpy as np

# 加载本地MNIST数据（假设路径正确）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建并训练模型（CPU版本）
model = models.Sequential([
    Input(shape=(28, 28)),  # 使用 Input 层指定输入形状
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))