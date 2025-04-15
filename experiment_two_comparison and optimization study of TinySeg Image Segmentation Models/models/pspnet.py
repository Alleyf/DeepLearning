import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)  # 获取特征图的高和宽
        pooled_features = [F.interpolate(stage(x), size=(h, w), mode='bilinear',
                                         align_corners=False) for stage in self.stages]
        pooled_features.append(x)  # 原始输入也加入特征拼接
        output = self.bottleneck(torch.cat(pooled_features, dim=1))  # 通道维度拼接
        return self.relu(self.bn(output))


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        # 使用 ResNet18 作为骨干网络
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # 去掉最后的 FC 和 Pooling
        # PSP 模块
        self.psp = PSPModule(in_channels=512, pool_sizes=[1, 2, 3, 6], out_channels=128)  # 输入通道 512 (ResNet18 输出维度)
        self.final = nn.Conv2d(128, num_classes, kernel_size=1)  # 最后分类器

    def forward(self, x):
        input_size = x.size()[2:]  # 获取输入尺寸 (H, W)
        x = self.encoder(x)  # 特征提取
        x = self.psp(x)  # 金字塔池化模块
        x = self.final(x)  # 分类部分
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)  # 动态调整到 128×128
        return x


def build_pspnet(num_classes):
    return PSPNet(num_classes)