import torch
from torch import nn
import torch.nn.functional as F


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
        # https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels*4, (21, 3), padding=(10, 4)) # Horizontal focus
        self.bn1 = nn.BatchNorm2d(out_channels*4)
        self.relu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(out_channels*4, out_channels*2, (3, 21), padding=(4, 10)) # Vertical focus
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(out_channels*2, out_channels, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

class NeuralNetwork(nn.Module):
    # Input is 512x512 grayscale image
    def __init__(self):
        super().__init__()
        self.stn = SpatialTransformer()  # Spatial transformer layer for handling rotation/perspective
        self.block1 = UNetBlock(1, 32)
        self.block2 = UNetBlock(32, 64)
        self.decoder = nn.Conv2d(64, 1, 5, padding=1)  # Output a binary mask

        self.final_activation = nn.Sigmoid()  # For generating a binary mask

    def forward(self, x):
        # Pass through spatial transformer (if needed)
        x = self.stn(x)

        # Encoder (downsampling)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        decoded = self.decoder(block2)

        # Apply sigmoid to get the final binary mask
        mask = self.final_activation(decoded)
        return mask
