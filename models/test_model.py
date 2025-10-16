"""import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")


class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        avg_out = avg_out.view(x.size(0), x.size(1), 1, 1)
        return x * avg_out


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, kernel_size=3, dilation=1, padding=1
    ):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.block = nn.Sequential(self.conv1, self.batch_norm1, self.relu)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, filters=128):
        super().__init__()
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        res = self.relu(self.conv(x))
        res = self.conv2(res)
        return self.relu2(res + x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=1,
        )
        self.relu = nn.LeakyReLU()
        self.block = nn.Sequential(self.conv1, self.relu)

    def forward(self, x):
        return self.block(x)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Encoder
        self.enc1 = EncoderBlock(
            1, 64, stride=1, kernel_size=(21, 9), padding=(10, 4)
        )  # 512x512 -> 512x512
        self.enc2 = EncoderBlock(
            64, 64, stride=2, kernel_size=(9, 21), padding=(4, 10)
        )  # 512x512 -> 256x256
        self.enc3 = EncoderBlock(
            64, 64, kernel_size=5, stride=2, padding=2
        )  # 256x256 -> 128x128

        # Bottleneck Residual Blocks
        self.residuals = nn.Sequential(
            ResidualBlock(64), ResidualBlock(64), ResidualBlock(64), CBAM(64)
        )

        # Decoder
        self.dec1 = DecoderBlock(64, 64, stride=2)  # 128x128 -> 256x256
        self.dec2 = DecoderBlock(64, 64, stride=2)  # 256x256 -> 512x512
        self.dec3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # 512x512 -> 512x512

    def forward(self, x):
        # Encoding
        # [-1, 1]
        x = (x - 0.5) / 0.5
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck
        x3 = self.residuals(x3)

        # Decoding
        x4 = self.dec1(x3) + x2
        x5 = self.dec2(x4) + x1

        out = torch.tanh(self.dec3(x5))
        # Rescale output to [0, 1] to match original image range
        out = (out + 1) / 2
        # Ensure x - out stays in range [0, 1]
        return torch.clamp(x - out, 0, 1)
"""

import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")


class LightweightCBAM(nn.Module):
    def __init__(self, channels):
        super(LightweightCBAM, self).__init__()
        # Simplified channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention only (more efficient)
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        channel_out = avg_out.view(x.size(0), x.size(1), 1, 1)
        return x * channel_out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightResBlock(nn.Module):
    def __init__(self, channels):
        super(LightResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=1 if stride > 1 else 0,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class NeuralNetwork(nn.Module):
    def __init__(self, base_channels=32):
        super(NeuralNetwork, self).__init__()

        # Encoder path - reduced channels
        self.enc1_h = EncoderBlock(
            1, base_channels, kernel_size=(11, 3), padding=(5, 1)
        )  # Horizontal lines
        self.enc1_v = EncoderBlock(
            1, base_channels, kernel_size=(3, 11), padding=(1, 5)
        )  # Vertical lines

        # Feature fusion with 1x1 conv
        self.fusion1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        # Downsampling with reduced parameters
        self.enc2 = EncoderBlock(
            base_channels, base_channels * 2, stride=2
        )  # 512x512 -> 256x256
        self.enc3 = EncoderBlock(
            base_channels * 2, base_channels * 2, stride=2
        )  # 256x256 -> 128x128

        # Lightweight bottleneck
        self.bottleneck = nn.Sequential(
            LightResBlock(base_channels * 2),
            LightResBlock(base_channels * 2),
            LightweightCBAM(base_channels * 2),
        )

        # Decoder path
        self.dec1 = DecoderBlock(
            base_channels * 2, base_channels * 2, stride=2
        )  # 128x128 -> 256x256
        self.dec2 = DecoderBlock(
            base_channels * 2, base_channels, stride=2
        )  # 256x256 -> 512x512

        # Skip connections with element-wise addition (lighter than concatenation)
        self.skip1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1)
        self.skip2 = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        # Final output layer
        self.final = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        # Simple line mask for adaptive blending
        self.line_mask = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        # Normalize input
        x_norm = (x - 0.5) / 0.5

        # Multi-directional feature extraction
        x1_h = self.enc1_h(x_norm)
        x1_v = self.enc1_v(x_norm)
        x1 = self.fusion1(torch.cat([x1_h, x1_v], dim=1))

        # Encoding
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck processing
        x3 = self.bottleneck(x3)

        # Decoding with lightweight skip connections
        x4 = self.dec1(x3)
        x4 = x4 + self.skip1(x2)  # Element-wise addition for skip connection

        x5 = self.dec2(x4)
        x5 = x5 + self.skip2(x1)  # Element-wise addition for skip connection

        # Line mask for adaptive processing
        mask = self.line_mask(x5)

        # Final output
        out = torch.tanh(self.final(x5))
        out = (out + 1) / 2  # Scale to [0, 1]

        # Blend original and processed image using the mask
        # Where mask=1 (detected lines), use the processed output
        # Where mask=0 (no lines), keep the original image
        result = x * (1 - mask) + out * mask

        return torch.clamp(result, 0, 1)
