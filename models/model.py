import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 512x512 -> 512x512
        self.enc2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # 512x512 -> 256x256
        self.enc3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )  # 256x256 -> 128x128

        # Bottleneck Residual Blocks
        self.res1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 128x128 -> 256x256
        self.dec2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 256x256 -> 512x512
        self.dec3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # 512x512 -> 512x512

    def forward(self, x):
        # Encoding
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))

        # Bottleneck
        x3 = F.relu(self.res1(x3)) + x3
        x3 = F.relu(self.res2(x3)) + x3

        # Decoding
        x4 = F.relu(self.dec1(x3)) + x2
        x5 = F.relu(self.dec2(x4)) + x1
        out = self.dec3(x5)  # Sigmoid for pixel output (0-1 range)

        return out


class NeuralNetworkLight(nn.Module):
    # Input is 512x512 grayscale image
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Rectangular filter: width = 41, height = 17
            nn.Conv2d(1, 32, (21, 9), padding=(10, 4)),  # (41-1)/2 Horizontal focus
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # RectFilter Transpose: width = 9, height = 21
            nn.Conv2d(32, 16, (9, 21), padding=(4, 10)),  # Vertical focus
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # Standard small filter for fine details
            nn.Conv2d(16, 8, (3, 3), padding=1),  # Small fine-grained filter
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        mask = self.network(x)  # Pass input through network to get the mask
        return mask
