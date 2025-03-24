import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
from albumentations import Compose, Resize, HorizontalFlip, RandomCrop, RandomBrightnessContrast, Normalize
from albumentations.pytorch import ToTensorV2


class VOCDataset(Dataset):
    def __init__(self, root_dir, image_set, image_size=(256, 256), augmentations=None):
        """
        初始化 VOC 数据集
        :param root_dir: 数据集根目录
        :param image_set: 数据划分 (train 或 val)
        :param image_size: 图像大小
        :param augmentations: 数据增强
        """
        if isinstance(root_dir, list):
            raise TypeError(f"root_dir 应是字符串，但收到 {type(root_dir)}")

        self.root_dir = root_dir
        self.image_set = image_set
        self.image_size = image_size
        self.augmentations = augmentations

        # 图片和掩码目录
        self.image_dir = os.path.join(self.root_dir, "JPEGImages")
        self.mask_dir = os.path.join(self.root_dir, "Annotations")

        # 数据划分文件
        self.set_file = os.path.join(self.root_dir, "ImageSets", f"{self.image_set}.txt")
        if not os.path.exists(self.set_file):
            raise FileNotFoundError(f"划分文件 {self.set_file} 不存在！")

        with open(self.set_file, "r") as f:
            self.image_ids = f.read().strip().split("\n")

        # 默认的图像 resize
        if not self.augmentations:
            self.augmentations = Resize(height=image_size[0], width=image_size[1])

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.image_ids)

    # def __getitem__(self, idx):
    #     """
    #     根据索引获取样本，包括图像和目标掩码
    #     """
    #     # 获取图像和掩码路径
    #     image_id = self.image_ids[idx]
    #     image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
    #     mask_path = os.path.join(self.mask_dir, f"{image_id}.png")
    #
    #     # 检查文件是否存在
    #     if not os.path.exists(image_path):
    #         raise FileNotFoundError(f"图像文件 {image_path} 找不到！")
    #     if not os.path.exists(mask_path):
    #         raise FileNotFoundError(f"掩码文件 {mask_path} 找不到！")
    #
    #         # 加载图像和掩码
    #     image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)  # [H, W, C]
    #     mask = np.array(Image.open(mask_path), dtype=np.uint8)  # [H, W]
    #
    #     # 如果定义了增强操作，则同时应用到图像和掩码
    #     if self.augmentations:
    #         augmented = self.augmentations(image=image, mask=mask)
    #         image = augmented["image"]
    #         mask = augmented["mask"]
    #
    #         # 转换为 PyTorch 的 Tensor 格式（仅当 image 是 np.ndarray 时）
    #     if isinstance(image, np.ndarray):
    #         image = torch.from_numpy(image).permute(2, 0, 1).float()  # [H, W, C] -> [C, H, W]
    #     if isinstance(mask, np.ndarray):
    #         mask = torch.from_numpy(mask).long()  # [H, W]
    #
    #     # 增加断言，确保格式正确
    #     assert image.shape[0] == 3, f"图像的通道数应为 3，但得到 {image.shape[0]}"
    #     assert len(mask.shape) == 2, f"掩码应为二维数组，但得到 {mask.shape}"
    #
    #     return image, mask
    def __getitem__(self, idx):
        """
        根据索引获取样本，包括图像和目标掩码
        """
        # 获取图像和掩码路径
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")

        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件 {image_path} 不存在！")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩码文件 {mask_path} 不存在！")

            # 加载图像和掩码
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)  # [H, W, C]
        mask = np.array(Image.open(mask_path), dtype=np.uint8)  # [H, W]

        # 如果定义了增强操作，则同时应用到图像和掩码
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            mask = mask.to(torch.int64)
            # 转换为 PyTorch Tensor 格式
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # [H, W, C] -> [C, H, W]
        if isinstance(mask, np.uint8):
            mask = mask.to(torch.int64)  # 确保掩码为 Long 类型
        # 增加断言，确保格式正确
        assert isinstance(image, torch.Tensor), f"图像应为 Tensor 类型，但得到 {type(image)}"
        assert isinstance(mask, torch.Tensor), f"掩码应为 Tensor 类型，但得到 {type(mask)}"
        assert image.shape[0] == 3, f"图像的通道数应为 3，但得到 {image.shape[0]}"
        assert len(mask.shape) == 2, f"掩码应为二维数组，但得到形状 {mask.shape}"

        return image, mask
def get_dataloader(
    root_dir, image_set="train", batch_size=8, image_size=(256, 256), augmentations=None, shuffle=True
):
    """
    获取 PyTorch DataLoader
    :param root_dir: 数据集根目录路径
    :param image_set: 数据集划分文件名（如 "train" 或 "val"）
    :param batch_size: 加载批大小
    :param image_size: 图像大小 (H, W)
    :param augmentations: 数据增强（albumentations）
    :param shuffle: 数据是否随机打乱
    :return: PyTorch 数据加载器
    """
    dataset = VOCDataset(
        root_dir=root_dir,
        image_set=image_set,
        image_size=image_size,
        augmentations=augmentations,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


def get_augmentations(image_size=(256, 256), mode="train"):
    """
    定义 albumentations 数据增强操作
    :param image_size: 图像大小 (H, W)
    :param mode: 数据模式 ("train" 或 "val")
    :return: albumentations.Compose 对象
    """
    if mode == "train":
        return Compose([
            Resize(height=image_size[0], width=image_size[1]),  # 调整大小
            HorizontalFlip(p=0.5),  # 水平翻转
            RandomCrop(height=image_size[0], width=image_size[1], p=0.5),  # 随机裁剪
            RandomBrightnessContrast(p=0.2),  # 随机调整亮度和对比度
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),  # 标准化
            ToTensorV2()  # 转为 PyTorch Tensor
        ])
    else:
        # 验证集仅处理为标准化和 Tensor 转换
        return Compose([
            Resize(height=image_size[0], width=image_size[1]),  # 调整大小
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
            ToTensorV2()
        ])


# 自动划分 train.txt 和 val.txt
def create_imageset_split(image_dir, output_dir, val_split=0.2):
    """
    自动生成 train.txt 和 val.txt 文件
    :param image_dir: JPEGImages 文件夹路径
    :param output_dir: ImageSets 文件夹路径
    :param val_split: 验证集比例
    """
    from sklearn.model_selection import train_test_split

    # 获取 JPEGImages 文件夹中的所有图片 ID（不带扩展名）
    image_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # 按验证集比例划分训练集和验证集
    train_ids, val_ids = train_test_split(image_ids, test_size=val_split, random_state=42)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 写入 train.txt 和 val.txt 文件
    with open(os.path.join(output_dir, "train.txt"), "w") as train_file:
        train_file.write("\n".join(train_ids) + "\n")
    with open(os.path.join(output_dir, "val.txt"), "w") as val_file:
        val_file.write("\n".join(val_ids) + "\n")

    print(f"划分文件已生成: {output_dir}/train.txt 和 {output_dir}/val.txt")


# 测试代码
if __name__ == "__main__":
    root_dir = r"E:\BaiduNetdiskDownload\tiny_seg_exp\tiny_seg_exp\TinySeg"  # 替换成你的路径

    # 自动生成划分文件（如果没有 train.txt 和 val.txt）
    images_dir = os.path.join(root_dir, "JPEGImages")
    imageset_dir = os.path.join(root_dir, "ImageSets")
    create_imageset_split(images_dir, imageset_dir, val_split=0.2)

    # 创建数据加载器
    train_loader = get_dataloader(
        root_dir=root_dir,
        image_set="train",
        batch_size=16,
        image_size=(256, 256),
        augmentations=get_augmentations(image_size=(256, 256), mode="train"),
        shuffle=True,
    )

    # 测试 DataLoader 输出
    for images, masks in train_loader:
        print("批图像形状:", images.shape)  # 形状: [batch_size, 3, 128, 128]
        print("批掩码形状:", masks.shape)  # 形状: [batch_size, 128, 128]
        print("掩码类别值:", torch.unique(masks))  # 检查 mask 中的类别值是否正确分类索引
        break