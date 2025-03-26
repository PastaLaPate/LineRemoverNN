import torch
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
        self.relu = nn.ReLU()
        self.block = nn.Sequential(self.conv1, self.batch_norm1, self.relu)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, filters=128):
        super().__init__()
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

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
        self.relu = nn.ReLU()
        self.block = nn.Sequential(self.conv1, self.relu)

    def forward(self, x):
        return self.block(x)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Encoder
        self.enc1 = EncoderBlock(
            1, 32, stride=1, kernel_size=3, padding=1
        )  # 512x512 -> 512x512
        self.enc2 = EncoderBlock(
            32, 64, stride=2, kernel_size=7, padding=3
        )  # 512x512 -> 256x256
        self.enc3 = EncoderBlock(
            64, 128, kernel_size=5, stride=2, dilation=2, padding=4
        )  # 256x256 -> 128x128

        # Bottleneck Residual Blocks
        self.residuals = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128), CBAM(128)
        )

        # Decoder
        self.dec1 = DecoderBlock(128, 64, stride=2)  # 128x128 -> 256x256
        self.dec2 = DecoderBlock(64, 32, stride=2)  # 256x256 -> 512x512
        self.dec3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # 512x512 -> 512x512

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
        # [-1, 1]
        out = torch.tanh(self.dec3(x5))
        # Rescale output to [0, 1] to match original image range
        out = (out + 1) / 2
        # Ensure x - out stays in range [0, 1]
        return torch.clamp(x - out, 0, 1)
