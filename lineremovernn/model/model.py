import torch
import torch.nn as nn
from torch import Tensor


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LineRemovalUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # --- Encoder (Downsampling) ---
        # Channels: 1 -> 32 -> 64 -> 128 -> 256
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(256, 512)

        # --- Decoder (Upsampling) ---
        # Note: input channels to decoder blocks are doubled because of the skip connections
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)  # 256 (skip) + 256 (up) = 512

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 (skip) + 128 (up) = 256

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)  # 64 (skip) + 64 (up) = 128

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)  # 32 (skip) + 32 (up) = 64

        # --- Final Mask Output ---
        self.mask_out = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # --- Encoder Pass ---
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # --- Bottleneck ---
        b = self.bottleneck(p4)

        # --- Decoder Pass with Skip Connections ---
        d4 = self.up4(b)
        d4 = torch.cat([e4, d4], dim=1)  # Concatenate spatial information from encoder
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        # --- The Residual Masking Step ---
        # 1. Predict the intensity of the lines
        mask_logits = self.mask_out(d1)
        line_mask = self.sigmoid(mask_logits)

        # 2. Subtract the mask (or add it, depending on normalization)
        # Assuming your input data is normalized to [0.0, 1.0] where 0.0 is black ink and 1.0 is white paper.
        # Since lines darken the page, we ADD the predicted line mask to restore the paper to white.
        clean_output = torch.clamp(x + line_mask, 0.0, 1.0)

        return clean_output


# Quick sanity check
if __name__ == "__main__":
    model = LineRemovalUNet().cuda()
    dummy_input = torch.randn(8, 1, 256, 256).cuda()  # Batch of 8, 1-channel, 256x256
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
