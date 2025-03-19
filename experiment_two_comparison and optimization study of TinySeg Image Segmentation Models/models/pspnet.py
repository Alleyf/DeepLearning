import torch
import torch.nn as nn
from torchvision.models import resnet18

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
            ) for size in pool_sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1)

    def forward(self, x):
        h, w = x.size()[2:]
        pyramids = [x]
        pyramids.extend([
            F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) 
            for stage in self.stages
        ])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.psp = PSPModule(512)
        self.final = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(size=128, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.psp(x)
        return self.final(x)