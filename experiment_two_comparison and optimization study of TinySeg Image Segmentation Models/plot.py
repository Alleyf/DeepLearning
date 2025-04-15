import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch


# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, train_miou, val_miou):
    """
    绘制训练曲线，包括损失曲线和mIoU曲线
    :param train_losses: 每个epoch的训练损失
    :param val_losses: 每个epoch的验证损失
    :param train_miou: 每个epoch的训练集mIoU
    :param val_miou: 每个epoch的验证集mIoU
    """
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 绘制mIoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_miou, label="Training mIoU", color="blue")
    plt.plot(epochs, val_miou, label="Validation mIoU", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.title("mIoU Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    绘制混淆矩阵 (Confusion Matrix)
    :param y_true: 真实标签 (B, H, W) 张量或 NumPy 数组
    :param y_pred: 预测标签 (B, H, W) 张量或 NumPy 数组
    :param class_labels: 类别名列表 ['class_0', 'class_1', ...]
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    cm = confusion_matrix(y_true, y_pred, normalize="true")  # 归一化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # 示例：训练过程可视化

    # 假设的训练记录
    train_losses = [1.15, 0.98, 0.89, 0.78, 0.65]  # 每个 epoch 的训练损失
    val_losses = [1.20, 1.05, 0.95, 0.85, 0.80]    # 每个 epoch 的验证损失
    train_miou = [0.50, 0.58, 0.60, 0.65, 0.70]    # 每个 epoch 的训练集 mIoU
    val_miou = [0.45, 0.50, 0.55, 0.60, 0.65]      # 每个 epoch 的验证集 mIoU

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_miou, val_miou)

    # 假设真实标签和预测标签
    y_true = torch.randint(0, 3, (10, 64, 64))  # 真实标签 (10 个样本，3 类)
    y_pred = torch.randint(0, 3, (10, 64, 64))  # 预测标签

    # 类别名称
    class_labels = ["Background", "Class_1", "Class_2"]

    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, class_labels)