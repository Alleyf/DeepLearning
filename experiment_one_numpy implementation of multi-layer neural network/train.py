import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import Layer, SoftmaxLayer, cross_entropy_loss
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from utils import TrainingUtils
from scipy.ndimage import rotate


# 数据加载与预处理
def load_data():
    """
    MNIST数据加载与预处理 (改进版)

    处理流程：
    1. 数据加载：使用sklearn的fetch_openml获取MNIST数据集
    2. 格式转换：将DataFrame转换为float32类型并归一化到[0,1]范围
    3. 标签处理：
       a. 转换为int8类型
       b. 使用OneHotEncoder进行稀疏矩阵编码
       c. 确保输出形状为(70000, 10)

    关键技术点：
    - parser='auto'自动选择最佳解析器
    - as_frame=True保持DataFrame格式便于后续处理
    - sparse_output=False生成密集矩阵加速计算
  
    步骤：
    1. 从OpenML加载MNIST数据集
    2. 像素值归一化到[0,1]范围
    3. 对标签进行One-hot编码

    返回：
        X: 归一化后的图像数据 (70000, 784)
        y_onehot: One-hot编码后的标签 (70000, 10)
    """
    # 使用parser='auto'和as_frame=True参数
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
    X = mnist.data.astype('float32') / 255.0

    # 确保target是正确的格式
    if hasattr(mnist.target, 'values'):
        y = mnist.target.astype('int8').values.reshape(-1, 1)
    else:
        y = mnist.target.astype('int8').reshape(-1, 1)

    # One-hot编码
    # One-hot编码器设置：生成密集矩阵格式（非稀疏矩阵）
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    return X, y_onehot


# 数据增强
def data_augmentation(X, y, num_augmented=1):
    """
    数据增强处理流程：
    1. 遍历原始数据集中的每个样本
    2. 对每个样本生成num_augmented个增强版本
    3. 应用随机旋转增强（角度范围：±15度）
    4. 保持图像尺寸不变(reshape=False)

    参数说明：
    :param num_augmented: 每个样本的增强次数
    :return: 
        augmented_X: 增强后的特征矩阵 (N*(1+num_augmented), 784)
        augmented_y: 对应的标签矩阵

    注意：增强后的图像使用flatten()保持输入维度一致性
    """
    augmented_X = []
    augmented_y = []
    for i in range(len(X)):
        img = X.iloc[i].values.reshape(28, 28)
        label = y[i]
        augmented_X.append(img.flatten())
        augmented_y.append(label)
        for _ in range(num_augmented):
            angle = np.random.randint(-15, 15)
            rotated_img = rotate(img, angle, reshape=False)
            augmented_X.append(rotated_img.flatten())
            augmented_y.append(label)
    return np.array(augmented_X), np.array(augmented_y)


# 网络初始化
def initialize_network():
    """
    网络结构维度说明：
    ----------------------------------------------------------------
    层序 | 类型      | 输入维度 | 输出维度 | 激活函数 | Dropout率
    ----------------------------------------------------------------
    1    | Dense    | 784     | 1024    | ReLU    | 0.3
    2    | Dense    | 1024    | 512     | ReLU    | 0.3
    3    | Dense    | 512     | 256     | ReLU    | 0.3
    4    | Softmax  | 256     | 10      | -       | -
    ----------------------------------------------------------------

    设计原则：
    - 每层神经元数量按~50%递减防止信息瓶颈
    - 使用Dropout缓解过拟合
    - 最后一层使用Softmax产生概率分布

    网络结构：
    - 输入层: 784神经元（对应28x28像素）
    - 隐藏层1: 1024神经元 + ReLU激活 + Dropout(0.3)
    - 隐藏层2: 512神经元 + ReLU激活 + Dropout(0.3)
    - 隐藏层3: 256神经元 + ReLU激活 + Dropout(0.3)
    - 输出层: 10神经元 + Softmax
    """
    layers = [
        Layer(784, 1024, 'relu', dropout_rate=0.3),
        Layer(1024, 512, 'relu', dropout_rate=0.3),
        Layer(512, 256, 'relu', dropout_rate=0.3),
        SoftmaxLayer(256, 10)
    ]
    return layers


# 训练循环
def train():
    """
    神经网络训练主循环

    超参数说明：
    - epochs: 训练轮次，控制遍历整个数据集的次数
    - batch_size: 128，使用小批量梯度下降优化
    - learning_rate: 0.0005，控制参数更新步长
    - reg_lambda: 0.001，L2正则化系数防止过拟合

    训练流程：
    1. 前向传播计算预测值
    2. 计算交叉熵损失（含L2正则化）
    3. 反向传播更新权重
    4. 每轮结束后验证集评估
    5. 保存最佳模型参数
    """
    # 初始化训练记录
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 超参数
    epochs = 30
    batch_size = 128
    learning_rate = 0.0005
    reg_lambda = 0.0001

    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据增强
    X_train, y_train = data_augmentation(X_train, y_train, num_augmented=1)

    layers = initialize_network()
    best_acc = 0.0

    # 初始化Adam优化器参数
    t = 0
    m_params = [np.zeros_like(layer.W) for layer in layers]
    v_params = [np.zeros_like(layer.W) for layer in layers]
    b_m_params = [np.zeros_like(layer.b) for layer in layers]
    b_v_params = [np.zeros_like(layer.b) for layer in layers]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for epoch in range(epochs):
        # 训练阶段
        for i in range(0, len(X_train), batch_size):
            # 前向传播
            X_batch = X_train[i:i + batch_size]
            a = X_batch
            for layer in layers:
                a = layer.forward(a)

            # 计算损失
            loss = cross_entropy_loss(a, y_train[i:i + batch_size], reg_lambda, layers)

            # 反向传播（基于链式法则）
            # 损失函数对输出的导数：∂L/∂a = (predicted - true)/batch_size
            # 逐层计算梯度：
            # ∂L/∂W = (a_prev.T @ δ) + λ*W （L2正则项）
            # ∂L/∂b = mean(δ, axis=0)
            # δ_{l-1} = δ @ W.T ⊙ σ'(z_{l-1})
            dW_list = []
            db_list = []

            # 输出层反向传播
            grad, dW, db = layers[-1].backward(y_train[i:i + batch_size], learning_rate, reg_lambda)
            dW_list.append(dW)
            db_list.append(db)

            # 隐藏层反向传播
            for layer in reversed(layers[:-1]):
                grad, dW, db = layer.backward(grad, learning_rate, reg_lambda)
                dW_list.insert(0, dW)
                db_list.insert(0, db)

            # Adam参数更新（自适应矩估计优化器）
        # 更新公式：
        # m_t = β1*m_{t-1} + (1-β1)*∇W
        # v_t = β2*v_{t-1} + (1-β2)*(∇W)^2
        # W = W - α*m̂_t/(√v̂_t + ε)
        # 其中 m̂_t = m_t/(1-β1^t), v̂_t = v_t/(1-β2^t)
            t += 1
            for i, layer in enumerate(layers):
                # 更新权重
                m_params[i] = beta1 * m_params[i] + (1 - beta1) * dW_list[i]
                v_params[i] = beta2 * v_params[i] + (1 - beta2) * (dW_list[i] ** 2)
                m_hat = m_params[i] / (1 - beta1 ** t)
                v_hat = v_params[i] / (1 - beta2 ** t)
                layer.W -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)  # 应用Adam更新规则：W = W - α*m̂/(√v̂+ε)

                # 更新偏置
                b_m_params[i] = beta1 * b_m_params[i] + (1 - beta1) * db_list[i]
                b_v_params[i] = beta2 * b_v_params[i] + (1 - beta2) * (db_list[i] ** 2)
                b_m_hat = b_m_params[i] / (1 - beta1 ** t)
                b_v_hat = b_v_params[i] / (1 - beta2 ** t)
                layer.b -= learning_rate * b_m_hat / (np.sqrt(b_v_hat) + eps)  # 偏置项更新：b = b - α*b_m̂/(√b_v̂+ε)

        # 验证阶段
        if hasattr(X_val, 'values'):
            a = X_val.values
        else:
            a = X_val
        for layer in layers:
            a = layer.forward(a)
        val_output = a
        val_acc = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))
        val_loss = cross_entropy_loss(val_output, y_val, reg_lambda, layers)

        # 记录指标
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f'Epoch {epoch + 1}/{epochs} => Train loss: {loss:.4f} | Val loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型参数（包含权重和偏置）
            params_dict = {}
            TrainingUtils.save_model(layers, params_dict)

    # 可视化训练过程
    TrainingUtils.plot_training_metrics(train_losses, val_losses, val_accuracies)
    TrainingUtils.plot_confusion_matrix(val_output, y_val)
    # 修改错误样本可视化逻辑
    val_pred = np.argmax(val_output, axis=1)
    val_true = np.argmax(y_val, axis=1)
    error_indices = np.where(val_pred != val_true)[0]
    np.random.shuffle(error_indices)
    sample_errors = error_indices[:16]

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(sample_errors):
        plt.subplot(4, 4, i + 1)
        if isinstance(X_val, pd.DataFrame):
            plt.imshow(X_val.iloc[idx].values.reshape(28, 28), cmap='gray')
        else:
            plt.imshow(X_val[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {val_true[idx]}\nPred: {val_pred[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('error_samples.png')
    plt.close()


def hyperparameter_search():
    """
    超参数敏感性分析实验（网格搜索法）

    实验设计：
    1. 全因子设计：学习率 × 优化器组合
    2. 评估指标：验证集准确率、收敛速度
    3. 早停策略：
       - 耐心窗口：3个epoch
       - 最小提升阈值：0.001

    结果分析：
    1. 保存不同参数组合的最终验证精度
    2. 记录达到最佳精度的收敛epoch数
    3. 可视化学习率与优化器的交互影响

    测试组合：
    - 学习率：[0.1, 0.01, 0.001]
    - 优化器：['sgd', 'adam']

    早停策略：
    - patience: 连续3轮验证准确率无提升则判定为收敛
    - min_delta: 判定准确率提升的最小阈值为0.001
    """
    # 早停策略参数
    patience = 3
    min_delta = 0.001
    import pandas as pd
    from itertools import product

    # 定义超参数
    batch_size = 128
    reg_lambda = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'optimizer': ['sgd', 'adam']
    }

    results = []

    for params in product(*param_grid.values()):
        lr, opt = params
        print(f'\n=== Running experiment with lr={lr}, optimizer={opt} ===')

        # 复制原始训练流程
        X, y = load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        # 数据增强
        X_train, y_train = data_augmentation(X_train, y_train, num_augmented=1)
        layers = initialize_network()

        # 修改训练参数
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        patience_counter = 0
        converged_epoch = 0

        for epoch in range(30):
            # 训练步骤
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                a = X_batch
                for layer in layers:
                    a = layer.forward(a)

                # 反向传播（基于链式法则）
            # 损失函数对输出的导数：∂L/∂a = (predicted - true)/batch_size
            # 逐层计算梯度：
            # ∂L/∂W = (a_prev.T @ δ) + λ*W （L2正则项）
            # ∂L/∂b = mean(δ, axis=0)
            # δ_{l-1} = δ @ W.T ⊙ σ'(z_{l-1})
                dW_list = []
                db_list = []
                grad, dW, db = layers[-1].backward(y_train[i:i + batch_size], lr, reg_lambda)
                dW_list.append(dW)
                db_list.append(db)

                for layer in reversed(layers[:-1]):
                    grad, dW, db = layer.backward(grad, lr, reg_lambda)
                    dW_list.insert(0, dW)
                    db_list.insert(0, db)

                # 参数更新
                if opt == 'sgd':
                    for j, layer in enumerate(layers):
                        layer.W -= lr * dW_list[j]
                        layer.b -= lr * db_list[j]
                elif opt == 'adam':
                    t = 0
                    m_params = [np.zeros_like(layer.W) for layer in layers]
                    v_params = [np.zeros_like(layer.W) for layer in layers]
                    b_m_params = [np.zeros_like(layer.b) for layer in layers]
                    b_v_params = [np.zeros_like(layer.b) for layer in layers]

                    t = 0  # 重置时间步计数器
                    for j, layer in enumerate(layers):
                        t += 1  # 每个参数单独更新时间步
                        # 更新权重参数
                        m_params[j] = beta1 * m_params[j] + (1 - beta1) * dW_list[j]
                        v_params[j] = beta2 * v_params[j] + (1 - beta2) * (dW_list[j] ** 2)
                        m_hat = m_params[j] / (1 - beta1 ** t)
                        v_hat = v_params[j] / (1 - beta2 ** t)
                        layer.W -= lr * m_hat / (np.sqrt(v_hat) + eps)

                        # 更新偏置参数
                        b_m_params[j] = beta1 * b_m_params[j] + (1 - beta1) * db_list[j]
                        b_v_params[j] = beta2 * b_v_params[j] + (1 - beta2) * (db_list[j] ** 2)
                        b_m_hat = b_m_params[j] / (1 - beta1 ** t)
                        b_v_hat = b_v_params[j] / (1 - beta2 ** t)
                        layer.b -= lr * b_m_hat / (np.sqrt(b_v_hat) + eps)

            # 验证阶段
            if hasattr(X_val, 'values'):
                a = X_val.values
            else:
                a = X_val
            for layer in layers:
                a = layer.forward(a)
            val_output = a
            val_acc = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))
            val_accuracies.append(val_acc)

            # 检查是否达到最佳准确率
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                converged_epoch = epoch
            else:
                patience_counter += 1

            # 如果连续patience轮没有改善，则认为已收敛
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        results.append({
            'learning_rate': lr,
            'optimizer': opt,
            'final_val_acc': val_accuracies[-1],
            'convergence_epoch': np.argmax(val_accuracies)
        })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv('hyperparameter_results.csv', index=False)

    # 绘制学习率对比曲线
    plt.figure(figsize=(10, 6))
    for opt in ['sgd', 'adam']:
        subset = df[df.optimizer == opt]
        plt.plot(subset.learning_rate, subset.final_val_acc, 'o-', label=opt)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Rate Sensitivity')
    plt.legend()
    plt.savefig('lr_sensitivity.png')
    plt.close()


if __name__ == '__main__':
    train()
    hyperparameter_search()