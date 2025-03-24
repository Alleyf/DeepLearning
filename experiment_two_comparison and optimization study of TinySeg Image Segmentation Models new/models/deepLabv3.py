import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        ])
        self.bottleneck = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        features = []
        for branch in self.branches:
            if isinstance(branch, nn.Sequential):  # 全局特征分支
                global_feature = branch(x)
                global_feature = F.interpolate(global_feature, size=x.shape[2:], mode='bilinear', align_corners=False)
                features.append(global_feature)
            else:
                features.append(branch(x))
        return self.bottleneck(torch.cat(features, dim=1))


class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.aspp = ASPPModule(in_channels=512, out_channels=128)
        self.final = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)  # 编码特征
        x = self.aspp(x)     # ASPP模块提取上下文信息
        x = self.final(x)    # 最终分类
        # 动态调整输出尺寸为输入图像的大小
        x = F.interpolate(x, size=(x.shape[2] * 32, x.shape[3] * 32), mode='bilinear', align_corners=False)
        return x


def build_deeplabv3(num_classes):
    return DeepLabV3(num_classes)


# 测试代码
if __name__ == "__main__":
    # 创建 DeepLabV3 模型
    model = build_deeplabv3(num_classes=6)
    # 构造一个模拟输入张量 (batch=2, channels=3, height=128, width=128)
    input_tensor = torch.randn(2, 3, 128, 128)
    # 前向传播
    output = model(input_tensor)
    # 打印输出张量的大小
    print("Output shape:", output.shape)