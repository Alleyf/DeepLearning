import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


def visualize_sample_comparison(dataset, pspnet_preds, deeplabv3_preds, ccnet_preds, num_classes, class_labels):
    """
    可视化样本对比：原图、真实掩码、PSPNet预测、DeepLabv3预测、CCNet预测
    :param dataset: 验证集样本 (list)，包含 (image, true_mask) 元组
    :param pspnet_preds: PSPNet 的预测结果 (B, H, W)
    :param deeplabv3_preds: DeepLabv3 的预测结果 (B, H, W)
    :param ccnet_preds: CCNet 的预测结果 (B, H, W)
    :param num_classes: 类别总数
    :param class_labels: 类别名称列表
    """
    n_samples = 3  # 每个类别挑选3张样本
    for class_idx in range(num_classes):
        fig, axes = plt.subplots(n_samples, 5, figsize=(15, 4 * n_samples))

        print(f"Visualizing class {class_labels[class_idx]}...")
        sample_count = 0
        for i, (image, true_mask) in enumerate(dataset):
            if sample_count >= n_samples:  # 每个类别仅展示3张样本
                break

            # 检查当前样本是否包含目标类别
            if not (true_mask == class_idx).any():
                continue

            # 获取预测结果
            pred_pspnet = pspnet_preds[i]
            pred_deeplabv3 = deeplabv3_preds[i]
            pred_ccnet = ccnet_preds[i]

            # 绘制每张样本
            axes[sample_count, 0].imshow(image.transpose(1, 2, 0))  # 原始图像
            axes[sample_count, 0].set_title("Original Image")
            axes[sample_count, 1].imshow(true_mask, cmap="tab20", vmin=0, vmax=num_classes)
            axes[sample_count, 1].set_title(f"Ground Truth ({class_labels[class_idx]})")
            axes[sample_count, 2].imshow(pred_pspnet, cmap="tab20", vmin=0, vmax=num_classes)
            axes[sample_count, 2].set_title("PSPNet Prediction")
            axes[sample_count, 3].imshow(pred_deeplabv3, cmap="tab20", vmin=0, vmax=num_classes)
            axes[sample_count, 3].set_title("DeepLabV3 Prediction")
            axes[sample_count, 4].imshow(pred_ccnet, cmap="tab20", vmin=0, vmax=num_classes)
            axes[sample_count, 4].set_title("CCNet Prediction")
            sample_count += 1

        plt.tight_layout()
        plt.show()


def visualize_error_analysis(image, true_mask, pred_mask, class_labels):
    """
    错误分析：标注假阳性（FP, 红色）与假阴性（FN, 蓝色）
    :param image: 输入图像 (C, H, W)
    :param true_mask: 真实掩码 (H, W)
    :param pred_mask: 预测掩码 (H, W)
    :param class_labels: 类别标签
    """
    fp = (pred_mask != true_mask) & (pred_mask > 0)  # 假阳性：预测中存在，真实中不存在
    fn = (pred_mask != true_mask) & (true_mask > 0)  # 假阴性：真实中存在，预测中不存在

    overlay_fp = np.zeros_like(image.transpose(1, 2, 0))  # 红色覆盖层
    overlay_fn = np.zeros_like(image.transpose(1, 2, 0))  # 蓝色覆盖层

    overlay_fp[fp.cpu().numpy()] = [1, 0, 0]  # 红色标注假阳性
    overlay_fn[fn.cpu().numpy()] = [0, 0, 1]  # 蓝色标注假阴性

    blended_image_fp = 0.5 * image.transpose(1, 2, 0) + 0.5 * overlay_fp
    blended_image_fn = 0.5 * image.transpose(1, 2, 0) + 0.5 * overlay_fn

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(image.transpose(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[1].imshow(true_mask, cmap="tab20", vmin=0, vmax=len(class_labels))
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred_mask, cmap="tab20", vmin=0, vmax=len(class_labels))
    axes[2].set_title("Predicted Mask")
    axes[3].imshow(blended_image_fp)
    axes[3].set_title("False Positives (Red)")
    axes[4].imshow(blended_image_fn)
    axes[4].set_title("False Negatives (Blue)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例输入数据
    num_classes = 3
    class_labels = ["Background", "Traffic Light", "Vegetation"]

    # 假设验证集数据（image, true_mask）
    dataset = [(torch.randint(0, 255, (3, 256, 256)), torch.randint(0, 3, (256, 256))) for _ in range(10)]

    # 模型预测（PSPNet, DeepLabV3, CCNet）
    pspnet_preds = [torch.randint(0, 3, (256, 256)) for _ in range(10)]
    deeplabv3_preds = [torch.randint(0, 3, (256, 256)) for _ in range(10)]
    ccnet_preds = [torch.randint(0, 3, (256, 256)) for _ in range(10)]

    # 可视化样本对比
    visualize_sample_comparison(dataset, pspnet_preds, deeplabv3_preds, ccnet_preds, num_classes, class_labels)

    # 可视化错误分析
    for image, true_mask in dataset[:3]:  # 示例3张样本
        pred_mask = torch.randint(0, 3, (256, 256))  # 模拟预测
        visualize_error_analysis(image, true_mask, pred_mask, class_labels)