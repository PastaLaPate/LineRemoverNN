import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),  # Padding adjusted
            nn.MaxPool2d(2, stride=2),  # Adjusted to stride=1 to preserve size
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),  # Padding adjusted
            nn.MaxPool2d(2, stride=2),  # Adjusted to stride=1 to preserve size
            nn.ReLU(True),
        )

        # Initialize fully connected layers with placeholder input size
        self.fc_loc = nn.Sequential(
            nn.Linear(
                10 * 124 * 124, 32
            ),  # Adjusted based on output size of the localization
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 124 * 124)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x


"""class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = SpatialTransformer()
        self.block1 = UNetBlock(
            1, 32, kernel_size=(21, 9), padding=(10, 4)
        )  # Adjusted padding
        self.block2 = UNetBlock(
            32, 16, kernel_size=(9, 21), padding=(4, 10)
        )  # Adjusted padding
        self.block3 = UNetBlock(
            16, 8, kernel_size=(3, 3), padding=1
        )  # Padding 1 for 3x3 kernel
        self.decoder = nn.Conv2d(
            8, 1, kernel_size=3, padding=1
        )  # Padding 1 to preserve size
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # x = self.stn(x)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        # decoded = self.decoder(block3)
        # mask = self.final_activation(decoded)
        return block3
"""

"""
class NeuralNetwork(nn.Module):
    # Input is 512x512 grayscale image
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Rectangular filter: width = 41, height = 17
            nn.Conv2d(1, 32, (31, 11), padding=(15, 5)),  # (41-1)/2 Horizontal focus
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # RectFilter Transpose: width = 9, height = 21
            nn.Conv2d(32, 16, (11, 31), padding=(5, 15)),  # Vertical focus
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # Standard small filter for fine details
            nn.Conv2d(16, 8, (7, 7), padding=3),  # Small fine-grained filter
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        mask = self.network(x)  # Pass input through network to get the mask
        return mask
"""


class NeuralNetwork(nn.Module):
    # Input is 512x512 grayscale image
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Rectangular filter: width = 41, height = 17
            nn.Conv2d(1, 32, (21, 9), padding=(10, 4)),  # (41-1)/2 Horizontal focus
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # RectFilter Transpose: width = 9, height = 21
            nn.Conv2d(32, 32, (9, 21), padding=(4, 10)),  # Vertical focus
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Standard small filter for fine details
            nn.Conv2d(32, 8, (3, 3), padding=1),  # Small fine-grained filter
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        mask = self.network(x)  # Pass input through network to get the mask
        return mask
