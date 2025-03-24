import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.ccnet import build_ccnet
from models.deepLabv3 import DeepLabV3
from models.pspnet import PSPNet
from dataloder import get_dataloader, get_augmentations
from utils import compute_pixel_accuracy, compute_miou, compute_dice_coefficient, compute_edge_f1_score

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_params(model):
    """获取模型参数量（百万）"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


def evaluate_fps(model, dataloader, num_iterations=50):
    """测试模型推理速度并计算 FPS"""
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_iterations:
                break
            images = images.to(device)
            _ = model(images)

    elapsed_time = time.time() - start_time
    avg_time_per_iter = elapsed_time / num_iterations
    fps = 1 / avg_time_per_iter
    return fps


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title="Confusion Matrix"):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def compute_edge_f1_score_with_fix(preds, targets):
    """
    修复后的计算边缘 F1 分数代码，确保 OpenCV 支持的数据输入
    """
    import cv2

    # 确保输入为 NumPy 数组
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

        # 如果输入是 batch 数据，取第一个样本作为代表
    if len(preds.shape) == 3:
        preds = preds[0]
    if len(targets.shape) == 3:
        targets = targets[0]

        # 转换为 OpenCV 支持的格式
    if preds.dtype != np.uint8:
        preds = (preds * 255).astype(np.uint8)
    if targets.dtype != np.uint8:
        targets = (targets * 255).astype(np.uint8)

        # 使用 Sobel 提取边缘
    preds_edge = cv2.Sobel(preds, cv2.CV_64F, 1, 1, ksize=3)
    targets_edge = cv2.Sobel(targets, cv2.CV_64F, 1, 1, ksize=3)

    # 可扩展：后续计算 F1 分数逻辑
    # 例如对边缘进行二值化处理再计算 F1
    f1_score = compute_edge_f1_score(preds_edge, targets_edge)
    return f1_score


def evaluate_model_with_ablation(model, dataloader, num_classes, classes):
    """
    消融实验评估：计算 mIoU 和边缘 F1，同时绘制混淆矩阵热力图
    """
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # 模型推理
            outputs = model(images)

            # 上采样到标签大小
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            # 合并数据
    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_targets = torch.cat(all_targets, dim=0).numpy().flatten()

    # 绘制混淆矩阵热力图
    plot_confusion_matrix(all_targets, all_preds, classes, normalize=True, title="Confusion Matrix - Ablation Study")

    # 计算 mIoU 和 边缘 F1
    miou = compute_miou(torch.tensor(all_preds).view(-1), torch.tensor(all_targets).view(-1), num_classes)
    edge_f1 = compute_edge_f1_score_with_fix(torch.tensor(all_preds).view(-1), torch.tensor(all_targets).view(-1))
    return miou, edge_f1


def evaluate_model_with_comparison(model, dataloader, num_classes):
    """
    模型评估：计算 mIoU、Dice 和模型参数量、FPS
    """
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # 模型推理
            outputs = model(images)

            # 上采样到标签大小
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    miou = compute_miou(all_preds, all_targets, num_classes)
    dice = compute_dice_coefficient(all_preds, all_targets, num_classes)
    return miou, dice


def load_model(model_name, variant):
    """加载模型及其权重"""
    weight_paths = {
        "ccnet": {
            "base": r"ccnet_base_aug_best.pth",
            "full_aug": r"ccnet_full_aug_best.pth",
            "no_aug": r"ccnet_no_aug_best.pth",
        },
        "deeplabv3": {
            "base": r"deeplabv3_base_aug_best.pth",
            "full_aug": r"deeplabv3_full_aug_best.pth",
            "no_aug": r"deeplabv3_no_aug_best.pth",
        },
        "pspnet": {
            "base": r"pspnet_base_aug_best.pth",
            "full_aug": r"pspnet_full_aug_best.pth",
            "no_aug": r"pspnet_no_aug_best.pth",
        },
    }
    weight_file = weight_paths[model_name][variant]
    if model_name == "ccnet":
        model = build_ccnet(num_classes=6)
    elif model_name == "deeplabv3":
        model = DeepLabV3(num_classes=6)
    elif model_name == "pspnet":
        model = PSPNet(num_classes=6)
    else:
        raise ValueError(f"未知模型名称：{model_name}")

    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # 数据路径与设置
    root_dir = r"E:\BaiduNetdiskDownload\tiny_seg_exp\tiny_seg_exp\TinySeg"
    num_classes = 6
    CLASSES = ["Background", "Class1", "Class2", "Class3", "Class4", "Class5"]
    augmentations = get_augmentations(image_size=(256, 256), mode="val")
    dataloader = get_dataloader(root_dir=root_dir, image_set="val", batch_size=4, augmentations=augmentations)

    # 模型和增强配置
    models = ["ccnet", "deeplabv3", "pspnet"]
    variants = ["base", "full_aug", "no_aug"]

    ablation_results = []
    comparison_results = []

    for model_name in models:
        for variant in variants:
            try:
                print(f"评估 {model_name} - {variant}...")
                model = load_model(model_name, variant)

                miou, edge_f1 = evaluate_model_with_ablation(model, dataloader, num_classes, CLASSES)
                ablation_results.append({
                    "Model": model_name,
                    "Variant": variant,
                    "mIoU": miou,
                    "Edge F1": edge_f1,
                })

                params = get_model_params(model)
                miou, dice = evaluate_model_with_comparison(model, dataloader, num_classes)
                fps = evaluate_fps(model, dataloader)
                comparison_results.append({
                    "Model": model_name,
                    "Variant": variant,
                    "Params (M)": params,
                    "mIoU": miou,
                    "Dice": dice,
                    "FPS": fps,
                })
            except Exception as e:
                print(f"评估失败 {model_name} - {variant}: {e}")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv("ablation_results.csv", index=False)
    print("数据增强消融实验结果已保存为 ablation_results.csv")

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv("comparison_results.csv", index=False)
    print("模型对比实验结果已保存为 comparison_results.csv")