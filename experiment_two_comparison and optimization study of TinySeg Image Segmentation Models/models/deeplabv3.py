import torch
import torch.nn as nn
from torchvision.models import resnet18

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = F.interpolate(self.global_pool(x), size=x.shape[2:], mode='bilinear')
        return self.fusion(torch.cat([x1, x2, x3, x4], dim=1))

class DeepLabv3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.aspp = ASPP(512)
        self.final = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(size=128, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.aspp(x)
        return self.final(x)