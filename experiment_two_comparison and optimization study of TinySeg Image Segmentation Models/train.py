import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import PSPNet, DeepLabv3, CCNet
from dataloader import get_dataloader
import numpy as np

# 设置随机种子
torch.manual_seed(42)

class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_miou = -np.inf

    def check_early_stop(self, val_miou):
        if val_miou > self.best_miou:
            self.best_miou = val_miou
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(model_name='PSPNet', num_classes=21, augmentation='full'):
    # 初始化模型
    models = {
        'PSPNet': PSPNet(num_classes),
        'DeepLabv3': DeepLabv3(num_classes),
        'CCNet': CCNet(num_classes)
    }
    model = models[model_name].cuda()

    # 获取数据加载器
    train_loader = get_dataloader(config=augmentation)
    val_loader = get_dataloader(config='no_aug')

    # 初始化优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 早停机制
    stopper = EarlyStopper()

    for epoch in range(50):
        # 训练阶段
        model.train()
        for images, masks in train_loader:
            # 前向传播和损失计算
            outputs = model(images.cuda())
            loss = criterion(outputs, masks.cuda())
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_miou = calculate_miou(model, val_loader)  # 需要实现mIoU计算
            
            if stopper.check_early_stop(val_miou):
                print(f'Early stopping at epoch {epoch}')
                break
        
        scheduler.step()

if __name__ == '__main__':
    train_model()