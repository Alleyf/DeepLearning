import numpy as np


class Layer:
    """
    全连接神经网络层

    参数：
        input_dim: 输入维度
        output_dim: 输出维度
        activation: 激活函数类型 ('relu','sigmoid','tanh')
        dropout_rate: Dropout丢弃率，范围[0,1]
    """

    def __init__(self, input_dim, output_dim, activation='relu', dropout_rate=0.0):
        # He初始化权重矩阵: W ~ N(0, sqrt(2/n)) * 0.01缩放
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        # 零初始化偏置向量 b ∈ R^(1×output_dim)
        self.b = np.zeros((1, output_dim))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.cache = None  # 缓存前向传播中间结果用于反向传播
        self.dropout_mask = None  # Dropout掩码

    def forward(self, X):
        """
        前向传播计算（含Dropout）

        参数：
            X: 输入矩阵 ∈ R^(m×input_dim)
        返回：
            a: 激活输出 ∈ R^(m×output_dim)

        计算步骤：
            1. z = X·W + b  线性变换
            2. a = σ(z)    激活函数
            3. Dropout: 训练时随机丢弃神经元
        """
        z = np.dot(X, self.W) + self.b  # 线性变换 z ∈ R^(m×output_dim)
        if self.activation == 'relu':
            a = self.relu(z)  # ReLU: max(0,z)
        elif self.activation == 'sigmoid':
            a = self.sigmoid(z)  # Sigmoid: 1/(1+e^{-z})
        elif self.activation == 'tanh':
            a = self.tanh(z)  # Tanh: (e^z - e^{-z})/(e^z + e^{-z})

        # 应用Dropout
        if self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
            a *= self.dropout_mask / (1 - self.dropout_rate)  # 缩放以保持期望值不变

        self.cache = (X, z, a)  # 缓存输入、线性输出、激活输出
        return a

    def backward(self, da, lr, reg_lambda):
        """
        反向传播计算梯度

        参数：
            da: 上游梯度 ∈ R^(m×output_dim)
            lr: 学习率
            reg_lambda: L2正则化系数

        返回：
            dX: 传递给下一层的梯度 ∈ R^(m×input_dim)

        梯度公式：
            dz = da ⊙ σ'(z)
            dW = (X.T·dz)/m + λW  (含L2正则化项)
            db = sum(dz)/m
            dX = dz·W.T
        """
        X, z, a = self.cache
        m = X.shape[0]  # 批大小

        # 计算本地梯度 dz
        if self.activation == 'relu':
            dz = da * self.relu_derivative(z)  # ReLU导数: 1(z>0)
        elif self.activation == 'sigmoid':
            dz = da * self.sigmoid_derivative(a)  # σ'(a) = a(1-a)
        elif self.activation == 'tanh':
            dz = da * self.tanh_derivative(a)  # tanh'(a) = 1 - a²

        # 计算参数梯度（含L2正则化）
        dW = np.dot(X.T, dz) / m + reg_lambda * self.W  # 正则化项 λW
        db = np.sum(dz, axis=0, keepdims=True) / m

        # 计算传递给前层的梯度
        dX = np.dot(dz, self.W.T)

        # 更新参数
        self.W -= lr * dW
        self.b -= lr * db
        return dX, dW, db

    @staticmethod
    def relu(x):
        """
        ReLU激活函数

        数学表达式:
            relu(x) = max(0, x)
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        ReLU导数

        计算规则:
            1. x > 0 时导数为1
            2. x <= 0 时导数为0
        """
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid激活函数

        数学表达式:
            σ(x) = 1 / (1 + e^{-x})
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(a):
        """
        Sigmoid导数

        已知激活输出a时，导数为:
            σ'(a) = a(1 - a)
        """
        return a * (1 - a)

    @staticmethod
    def tanh(x):
        """
        Tanh激活函数

        数学表达式:
            tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(a):
        """
        Tanh导数

        已知激活输出a时，导数为:
            tanh'(a) = 1 - a²
        """
        return 1 - a ** 2


class SoftmaxLayer:
    """
    Softmax输出层（配合交叉熵损失）

    参数：
        input_dim: 输入维度
        output_dim: 输出类别数
    """

    def __init__(self, input_dim, output_dim):
        # Xavier初始化权重矩阵
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        # 零初始化偏置
        self.b = np.zeros((1, output_dim))
        self.cache = None  # 缓存前向传播结果

    def forward(self, X):
        """
        Softmax前向传播（数值稳定实现）

        计算步骤：
            1. z = X·W + b
            2. 稳定化计算: exp(z - max(z))
            3. softmax: a = exp(z) / sum(exp(z))
        """
        z = np.dot(X, self.W) + self.b
        # 数值稳定处理：减去最大值防止指数爆炸
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(z_stable)
        a = exps / np.sum(exps, axis=1, keepdims=True)
        self.cache = (X, z, a)
        return a

    def backward(self, y_true, lr, reg_lambda):
        """
        Softmax层反向传播（配合交叉熵损失）

        梯度推导：
            dz = a - y_true （当使用交叉熵损失时）
            dW = X.T·dz / m + λW
            db = sum(dz) / m
            dX = dz·W.T
        """
        X, z, a = self.cache
        m = y_true.shape[0]

        # 交叉熵损失的梯度公式：dz = a - y_true
        dz = a - y_true

        # 计算参数梯度（含L2正则化）
        dW = np.dot(X.T, dz) / m + reg_lambda * self.W
        db = np.sum(dz, axis=0, keepdims=True) / m

        # 计算传递给前层的梯度
        dX = np.dot(dz, self.W.T)

        # 更新参数
        self.W -= lr * dW
        self.b -= lr * db
        return dX, dW, db


def cross_entropy_loss(y_pred, y_true, reg_lambda, layers):
    """
    计算交叉熵损失（含L2正则化）

    参数:
        y_pred: 模型预测概率分布
        y_true: 真实标签的one-hot编码
        reg_lambda: L2正则化系数
        layers: 所有需要正则化的网络层列表

    返回:
        total_loss: 交叉熵损失 + L2正则化项

    计算公式:
        CE = -1/m * Σ(y_true * log(y_pred))
        L2 = λ/(2m) * Σ(||W||²)
    """
    m = y_true.shape[0]
    ce_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # 添加1e-8防止log(0)
    l2_loss = 0
    for layer in layers:
        l2_loss += np.sum(np.square(layer.W))  # 累加所有权重的L2范数
    l2_loss = reg_lambda * l2_loss / (2 * m)  # 正则化项系数计算
    return ce_loss + l2_loss