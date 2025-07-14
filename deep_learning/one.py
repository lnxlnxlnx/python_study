import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 简单神经网络
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

# 使用示例
X = np.array([[0.1, 0.2]])
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
output = nn.forward(X)
print("Output:", output)
print("Network weights:", nn.W1, nn.W2)
print("Network biases:", nn.b1, nn.b2)
print("运行成功!!")  # 确认代码运行成功