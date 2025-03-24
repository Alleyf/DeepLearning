import os
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.pspnet import build_pspnet
from models.deepLabv3 import build_deeplabv3
from models.ccnet import build_ccnet
from dataloder import get_dataloader, create_imageset_split, get_augmentations
from evaluate import compute_miou
from utils import set_seed

    # 设置随机种子
set_seed(42)

    # 损失函数定义
criterion_ce = nn.CrossEntropyLoss()

def dice_loss(pred, target, smooth=1e-6):

        """
        Dice Loss 计算
        """
        pred = torch.softmax(pred, dim=1)
        target_one_hot = nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

def combined_loss(pred, target):
        """
        CrossEntropyLoss 和 DiceLoss 的加权组合
        """
        loss_ce = criterion_ce(pred, target)
        loss_dice = dice_loss(pred, target)
        return loss_ce + loss_dice

    # 加载训练和验证图片与掩码路径
def load_dataset_paths(root_dir):
        """
        根据数据集的目录结构，加载图片和掩码路径。
        root_dir: 数据集根目录包含 JPEGImages, Annotations 和 ImageSets 文件夹
        """
        images_dir = os.path.join(root_dir, "JPEGImages")
        masks_dir = os.path.join(root_dir, "Annotations")

        train_txt = os.path.join(root_dir, "ImageSets", "train.txt")
        val_txt = os.path.join(root_dir, "ImageSets", "val.txt")

        # 函数：从文本文件中获取图片文件名
        def read_file_list(txt_file):
            with open(txt_file, "r") as f:
                file_names = f.read().splitlines()
            return file_names

        train_file_names = read_file_list(train_txt)
        val_file_names = read_file_list(val_txt)

        # 拼接完整路径
        train_images = [os.path.join(images_dir, f"{name}.jpg") for name in train_file_names]
        train_masks = [os.path.join(masks_dir, f"{name}.png") for name in train_file_names]

        val_images = [os.path.join(images_dir, f"{name}.jpg") for name in val_file_names]
        val_masks = [os.path.join(masks_dir, f"{name}.png") for name in val_file_names]

        return train_images, train_masks, val_images, val_masks

    # 训练函数
def train_model(model, train_loader, val_loader, num_epochs=50,early_stop_patience=30, model_name="pspnet", aug_name="no_aug"):
        """
        模型训练函数
    :param model: 语义分割模型
    :param train_loader: 训练集数据加载器
    :param val_loader: 验证集数据加载器
    :param num_epochs: 训练轮数
    :param lr: 学习率
    :param weight_decay: 权重衰减
    :param early_stop_patience: 早停计数
    :param model_name: 模型架构名称
    :param aug_name: 数据增强策略名称
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 优化器和学习率调度器
        # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

        # 训练过程记录
        train_losses = []
        val_mious = []

        best_miou = 0.0
        stop_counter = 0

        # 动态生成模型权重文件名
        weight_file_name = f"{model_name}_{aug_name}_best2.pth"

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                #
                # print(f"Target min: {masks.min().item()}, Target max: {masks.max().item()}")
                # # 将负值映射为类别 0
                # masks[masks < 0] = 0
                # print(masks)
                optimizer.zero_grad()
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))

            # 验证阶段
            model.eval()
            val_miou = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    val_miou += compute_miou(preds, masks, num_classes=3)

            val_miou /= len(val_loader)
            val_mious.append(val_miou)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val mIoU: {val_miou:.4f}")

            # 早停机制
            if val_miou > best_miou:
                best_miou = val_miou
                stop_counter = 0
                torch.save(model.state_dict(), weight_file_name)
            else:
                stop_counter += 1
                if stop_counter >= early_stop_patience:
                    print("Early stopping triggered.")
                    break

            scheduler.step()

        # 绘制训练曲线
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_mious, label="Val mIoU")
        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.legend()

        plt.show()

    # 主函数
if __name__ == "__main__":
        # 设置根目录
        root_dir = r"E:\BaiduNetdiskDownload\tiny_seg_exp\tiny_seg_exp\TinySeg"

        # 自动生成 train.txt 和 val.txt（仅首次运行）
        images_dir = os.path.join(root_dir, "JPEGImages")
        imageset_dir = os.path.join(root_dir, "ImageSets")
        if not os.path.exists(os.path.join(imageset_dir, "train.txt")):
            create_imageset_split(images_dir, imageset_dir, val_split=0.2)

            # 数据增强配置
        from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, \
    ElasticTransform, GridDistortion, RandomCrop, Normalize

        # 数据增强配置
        # 仅归一化增强策略 (no_aug)
        no_aug = Compose([
            Normalize(mean=(0.485, 0.456, 0.406),  # 归一化均值 (ImageNet)
                      std=(0.229, 0.224, 0.225),  # 归一化标准差 (ImageNet)
                      max_pixel_value=255),  # 将像素值归一化到 [0, 1]
            ToTensorV2()  # 转为 PyTorch Tensor
        ])
        # 基础增强
        base_aug = Compose([
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.5),  # 随机旋转 [-15°, 15°]
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)
        ])
        # 完整增强
        advanced_aug = Compose([
            ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            RandomCrop(width=96, height=96, p=0.5)
        ])
        # 初始化数据增强策略
        aug_name = "full_aug"  # 可修改为 "base_aug" 或 "full_aug"
        if aug_name == "no_aug":
            augmentations = Compose(
                [Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255), ToTensorV2()])
        elif aug_name == "base_aug":
            augmentations = Compose([*base_aug.transforms, *base_aug.transforms])  # 归一化+基础增强
        elif aug_name == "full_aug":
            augmentations = Compose([*base_aug.transforms, *advanced_aug.transforms])  # 归一化 + 高级增强

        # 加载训练集和验证集的数据加载器
        train_loader = get_dataloader(
            root_dir=root_dir,  # 根目录是一个字符串，不是列表
            image_set="train",  # train 集划分名
            batch_size=512,
            image_size=(256, 256),
            augmentations=get_augmentations(image_size=(128, 128), mode="train"),
            shuffle=True
        )

        # 验证数据加载器
        val_loader = get_dataloader(
            root_dir=root_dir,
            image_set="val",
            batch_size=128,
            image_size=(256, 256),
            augmentations=get_augmentations(image_size=(128, 128), mode="val"),
            shuffle=False
        )
        # 模型选择
        model_name = "ccnet"  # 可切换deeplabv3 和 ccnet
        # model = build_pspnet(num_classes=6)  # build_pspnet
        # model = build_deeplabv3(num_classes=6)  # 可切换为 build_deeplabv3
        model = build_ccnet(num_classes=6)  # 可切换为  build_ccnet

        # 开始训练
        train_model(model, train_loader, val_loader, num_epochs=50,
                    model_name=model_name, aug_name=aug_name)