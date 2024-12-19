import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, x):
        # This layer would learn to apply a transformation (rotation, scaling, etc.)
        return F.grid_sample(x, self.get_transform_grid(x), padding_mode='border')

    def get_transform_grid(self, x):
        # This should return a transformation grid; for simplicity, let's assume no transformation
        # A better implementation would learn this transformation
        batch_size, _, height, width = x.size()
        return F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1), (batch_size, 1, height, width))

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (21, 3), padding=padding) # Horizontal focus
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 21), padding=padding) # Vertical focus
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.pool(x)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.block1 = UNetBlock(in_channels, out_channels, padding=1)
        self.block2 = UNetBlock(in_channels, out_channels, padding=1)

        self.upconv = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)

    def forward(self, x, skip_connection):
        x = self.block1(x)
        x = self.block2(x)
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)  # Skip connection for U-Net
        return x

class NeuralNetwork(nn.Module):
    # Input is 512x512 grayscale image
    def __init__(self):
        super().__init__()
        self.stn = SpatialTransformer()  # Spatial transformer layer for handling rotation/perspective
        self.encoder1 = UNetBlock(1, 32)
        self.encoder2 = UNetBlock(32, 64)
        self.encoder3 = UNetBlock(64, 128)

        self.decoder1 = UNetDecoderBlock(128 + 64, 64)
        self.decoder2 = UNetDecoderBlock(64 + 32, 32)
        self.decoder3 = nn.Conv2d(32, 1, 3, padding=1)  # Output a binary mask

        self.final_activation = nn.Sigmoid()  # For generating a binary mask

    def forward(self, x):
        # Pass through spatial transformer (if needed)
        x = self.stn(x)

        # Encoder (downsampling)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # Decoder (upsampling with skip connections)
        dec1 = self.decoder1(enc3, enc2)
        dec2 = self.decoder2(dec1, enc1)
        dec3 = self.decoder3(dec2)

        # Apply sigmoid to get the final binary mask
        mask = self.final_activation(dec3)
        return mask
