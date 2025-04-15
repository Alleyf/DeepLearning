import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# Criss-Cross Attention 模块
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        """
        Criss-Cross Attention 模块，支持水平和垂直方向的交叉注意力机制以及多头机制。
        """
        super(CrissCrossAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads!"
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads  # 每个头的维度

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习权重参数

    def forward(self, x):
        """
        前向传播：计算交叉注意力并输出加权结果
        """
        B, C, H, W = x.size()
        assert C == self.num_heads * self.head_dim, "Input channels must match num_heads * head_dim"

        # 生成 Query、Key 和 Value
        query = self.query_conv(x).view(B, self.num_heads, self.head_dim, H * W)  # B x num_heads x head_dim x HW
        key = self.key_conv(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # B x num_heads x HW x head_dim
        value = self.value_conv(x).view(B, self.num_heads, self.head_dim, H * W)  # B x num_heads x head_dim x HW

        # 计算注意力权重
        attention = torch.matmul(query, key)  # B x num_heads x HW x HW
        attention = F.softmax(attention, dim=-1)

        # 加权 Value 输出
        out = torch.matmul(attention, value)  # B x num_heads x head_dim x HW
        out = out.view(B, C, H, W)  # 还原形状

        # 融合结果
        return self.gamma * out + x


# CCNet 主体模型
class CCNet(nn.Module):
    def __init__(self, num_classes, num_heads=2):
        """
        CCNet 模型，基于 ResNet-18 编解码结构，结合 Criss-Cross Attention 模块。
        """
        super(CCNet, self).__init__()

        # 使用 ResNet-18 的前 4 个阶段作为编码器
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        # 添加 Criss-Cross Attention 模块
        self.cc_attention = CrissCrossAttention(in_channels=512, num_heads=num_heads)

        # 分类头：1×1 卷积用于特征映射到类别预测
        self.final = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        前向传播：输入图像 -> 编码器 -> Criss-Cross Attention -> 分类头
        """
        input_size = x.shape[2:]  # 获取输入图像的原始分辨率 (H, W)

        # 编码器和 CCAttention
        x = self.encoder(x)
        x = self.cc_attention(x)

        # 分类头
        x = self.final(x)

        # 动态调整分辨率到输入图像的分辨率
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


# 构建 CCNet 模型
def build_ccnet(num_classes, num_heads=2):
    """
    构建 CCNet 模型的静态方法
    """
    return CCNet(num_classes=num_classes, num_heads=num_heads)


# 测试 CCNet 模型
if __name__ == "__main__":
    # 定义模型
    model = build_ccnet(num_classes=6, num_heads=2)

    # 模拟输入数据
    input_tensor = torch.randn(2, 3, 128, 128)  # 输入大小为 (batch_size=2, channels=3, height=128, width=128)

    # 前向传播
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)  # 输出预测张量大小应为 [2, num_classes, 128, 128]