import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_baseline import DoubleConv, Down, OutConv


class AttentionGate(nn.Module):
    """利用解码器上下文对编码器特征进行空间筛选。"""

    def __init__(self, gating_channels, skip_channels, intermediate_channels):
        super().__init__()
        self.gating_projection = nn.Sequential(
            nn.Conv2d(gating_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, gating, skip):
        alpha = self.attention(
            self.gating_projection(gating) + self.skip_projection(skip)
        )
        return skip * alpha


class AttentionUp(nn.Module):
    """上采样解码器特征，过滤跳跃连接特征后再进行融合。"""

    def __init__(self, in_channels, out_channels, intermediate_channels):
        super().__init__()
        decoder_channels = in_channels // 2
        self.up = nn.ConvTranspose2d(
            in_channels, decoder_channels, kernel_size=2, stride=2
        )
        self.attention = AttentionGate(
            gating_channels=decoder_channels,
            skip_channels=decoder_channels,
            intermediate_channels=intermediate_channels,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, decoder_features, skip_features):
        decoder_features = self.up(decoder_features)

        # 与基线保持一致，兼容宽高不是 16 倍数的输入。
        diff_y = skip_features.size(2) - decoder_features.size(2)
        diff_x = skip_features.size(3) - decoder_features.size(3)
        decoder_features = F.pad(
            decoder_features,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        attended_skip = self.attention(decoder_features, skip_features)
        features = torch.cat([attended_skip, decoder_features], dim=1)
        return self.conv(features)


class AttentionUNet(nn.Module):
    """在全部编码器到解码器跳跃连接中加入注意力门的 U-Net。"""

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = AttentionUp(1024, 512, 256)
        self.up2 = AttentionUp(512, 256, 128)
        self.up3 = AttentionUp(256, 128, 64)
        self.up4 = AttentionUp(128, 64, 32)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    model = AttentionUNet()
    for height, width in [(256, 256), (255, 257)]:
        input_tensor = torch.randn(1, 3, height, width)
        output = model(input_tensor)
        assert output.shape == (1, 1, height, width)
        print(f"输入形状: {input_tensor.shape} -> 输出形状: {output.shape}")
    print("Attention U-Net 尺寸检查通过")
