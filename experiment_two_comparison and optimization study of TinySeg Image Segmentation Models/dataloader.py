from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class TinySegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augmentation='full'):
        import os
        
        # 扫描图像和掩码文件路径
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        # 初始化增强配置
        self.augmentation = augmentation
        
        # 定义基础增强
        self.base_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # 定义完整增强（基础+高级）
        self.full_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.RandomCrop(height=96, width=96, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        # 读取图像和掩码
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 应用数据增强
        if self.augmentation == 'full':
            transformed = self.full_aug(image=image, mask=mask)
        else:
            transformed = self.base_aug(image=image, mask=mask)
        
        return transformed['image'], transformed['mask']

    def __len__(self):
        # 返回数据集长度
        return len(self.image_paths)

def get_dataloader(config='no_aug', batch_size=16):
    # 根据配置返回对应的数据加载器
    transform = {
        'no_aug': A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'base_aug': TinySegDataset.base_aug,
        'full_aug': TinySegDataset.full_aug
    }[config]
    
    dataset = TinySegDataset(transform=transform)
    # 添加数据集划分功能
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_set, batch_size=batch_size)
    }