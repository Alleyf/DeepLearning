import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TrainingUtils:
    @staticmethod
    def load_data():
        """
        MNIST数据加载与预处理

        步骤：
        1. 从OpenML加载MNIST数据集
        2. 像素值归一化到[0,1]范围
        3. 对标签进行One-hot编码

        返回：
            X: 归一化后的图像数据 (70000, 784)
            y_onehot: One-hot编码后的标签 (70000, 10)
        """
        mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
        X = mnist.data.astype('float32') / 255.0

        if hasattr(mnist.target, 'values'):
            y = mnist.target.astype('int8').values.reshape(-1, 1)
        else:
            y = mnist.target.astype('int8').reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y)

        return X, y_onehot

    @staticmethod
    def evaluate(X, y, layers):
        """
        模型评估函数

        参数：
            X: 输入数据
            y: 真实标签的one-hot编码
            layers: 神经网络各层

        返回：
            分类准确率：预测正确的样本比例
        """
        a = X
        for layer in layers:
            a = layer.forward(a)
        predictions = np.argmax(a, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

    @staticmethod
    def save_model(layers, params_dict):
        # 确保文件保存在当前目录下
        save_path = os.path.join(CURRENT_DIR, 'best_model.npz')
        for i, layer in enumerate(layers):
            params_dict[f'layer{i}_W'] = layer.W
            params_dict[f'layer{i}_b'] = layer.b
        np.savez(save_path, **params_dict)

    @staticmethod
    def plot_training_metrics(train_losses, val_losses, val_accuracies):
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, 'g', label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(CURRENT_DIR, 'training_metrics.png')
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(val_output, y_val):
        val_pred = np.argmax(val_output, axis=1)
        val_true = np.argmax(y_val, axis=1)
        cm = confusion_matrix(val_true, val_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        save_path = os.path.join(CURRENT_DIR, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_error_samples(X_val, val_output, y_val):
        val_pred = np.argmax(val_output, axis=1)
        val_true = np.argmax(y_val, axis=1)
        error_indices = np.where(val_pred != val_true)[0]
        np.random.shuffle(error_indices)
        sample_errors = error_indices[:16]

        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(sample_errors):
            plt.subplot(4, 4, i + 1)
            if hasattr(X_val[idx], 'values'):
                plt.imshow(X_val[idx].values.reshape(28, 28), cmap='gray')
            else:
                plt.imshow(X_val[idx].reshape(28, 28), cmap='gray')
            plt.title(f'True: {val_true[idx]}\nPred: {val_pred[idx]}')
            plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join(CURRENT_DIR, 'error_samples.png')
        plt.savefig(save_path)
        plt.close()