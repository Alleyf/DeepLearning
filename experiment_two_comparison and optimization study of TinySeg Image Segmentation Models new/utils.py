import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# -----------------------------
# 随机种子设置
# -----------------------------
def set_seed(seed=42):
    """
    设置随机种子以确保实验结果可复现
    :param seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# -----------------------------
# 像素分类准确度
# -----------------------------
def compute_pixel_accuracy(preds, targets):
    """
    计算像素分类准确率
    :param preds: 预测结果 (H, W) 或 (B, H, W)
    :param targets: 标签 (H, W) 或 (B, H, W)
    :return: 像素分类准确率 (float)
    """
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

# -----------------------------
# mIoU 计算
# -----------------------------
def compute_miou(preds, targets, num_classes):
    """
    计算 mIoU (Mean Intersection over Union)
    :param preds: 模型预测 (B, H, W)
    :param targets: 真实标签 (B, H, W)
    :param num_classes: 类别数量
    :return: mIoU 值 (float)
    """
    iou_list = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        if union == 0:
            iou_list.append(1.0)  # 如果某类别不存在，mIoU 记为 1
        else:
            iou_list.append(intersection / union)
    return np.mean(iou_list)

# -----------------------------
# Dice 系数计算
# -----------------------------
def compute_dice_coefficient(preds, targets, num_classes):
    """
    计算 Dice 系数
    :param preds: 模型预测 (B, H, W)
    :param targets: 真实标签 (B, H, W)
    :param num_classes: 类别数量
    :return: 平均 Dice 系数 (float)
    """
    dice_list = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().item()
        total_area = (preds == cls).sum().item() + (targets == cls).sum().item()
        if total_area == 0:
            dice_list.append(1.0)
        else:
            dice_list.append(2 * intersection / total_area)
    return np.mean(dice_list)

# -----------------------------
# 边缘提取工具
# -----------------------------
def extract_edges(mask):
    """
    使用 Sobel 边缘检测提取单通道掩码的边缘
    :param mask: 掩码图像 (H, W)
    :return: 边缘掩码 (H, W)
    """
    # 检查数据形状
    if len(mask.shape) == 3:  # 如果输入是 3 通道图像 (H, W, C)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 确保数据类型
    if mask.dtype != np.uint8 and mask.dtype != np.float32:
        # print(f"警告：数据类型为 {mask.dtype}，已自动转换为 np.float32")
        mask = mask.astype(np.float32)

    # 使用 Sobel 算子提取边缘
    sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值（并二值化结果）
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = (edges > 0).astype(np.uint8)  # 二值化
    return edges

# -----------------------------
# 边缘 F1 分数计算
# -----------------------------
def compute_edge_f1_score(preds, targets):
    """
    计算边缘上的 F1 分数
    :param preds: 模型预测 (B, H, W)
    :param targets: 真实标签 (B, H, W)
    :return: 平均边缘 F1 分数 (float)
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    f1_scores = []
    for p, t in zip(preds, targets):
        pred_edge = extract_edges(p)
        target_edge = extract_edges(t)

        tp = np.logical_and(pred_edge, target_edge).sum()
        fp = np.logical_and(pred_edge, np.logical_not(target_edge)).sum()
        fn = np.logical_and(np.logical_not(pred_edge), target_edge).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f1_scores.append(f1)
    return np.mean(f1_scores)

# -----------------------------
# 绘制混淆矩阵
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    绘制混淆矩阵
    :param y_true: 真实标签 (H, W)
    :param y_pred: 模型预测 (H, W)
    :param class_labels: 类别名称列表
    """
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=list(range(len(class_labels))))
    cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # 归一化
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# -----------------------------
# 错误分析可视化
# -----------------------------
def visualize_error_analysis(image, true_mask, pred_mask):
    """
    可视化假阳性 (FP) 和假阴性 (FN)
    :param image: 原始图像 (H, W, C)
    :param true_mask: 真实标签 (H, W)
    :param pred_mask: 模型预测结果 (H, W)
    """
    fp = (pred_mask != true_mask) & (pred_mask > 0)  # 假阳性
    fn = (pred_mask != true_mask) & (true_mask > 0)  # 假阴性

    # 创建红色 (FP) 和蓝色 (FN) 的覆盖层
    overlay_fp = np.zeros_like(image, dtype=np.uint8)
    overlay_fn = np.zeros_like(image, dtype=np.uint8)
    overlay_fp[fp] = [255, 0, 0]  # 红色：假阳性
    overlay_fn[fn] = [0, 0, 255]  # 蓝色：假阴性

    blended_fp = cv2.addWeighted(image, 0.7, overlay_fp, 0.3, 0)
    blended_fn = cv2.addWeighted(image, 0.7, overlay_fn, 0.3, 0)

    # 绘制图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[1].imshow(blended_fp)
    axes[1].set_title("False Positives (Red)")
    axes[2].imshow(blended_fn)
    axes[2].set_title("False Negatives (Blue)")
    plt.tight_layout()
    plt.show()