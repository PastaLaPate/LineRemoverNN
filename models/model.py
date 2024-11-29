import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
class NeuralNetwork(nn.Module):
    # Input is 512x512 grayscale image
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Rectangular filter: width = 21, height = 9
            nn.Conv2d(1, 32, (21, 9), padding=(10, 4)),  # Horizontal focus
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # RectFilter Transpose: width = 9, height = 21
            nn.Conv2d(32, 16, (9, 21), padding=(4, 10)),  # Vertical focus
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            
            nn.Conv2d(16, 8, (3, 3), padding=1),   # Smaller filter, 16 -> 8 filters
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        mask = self.network(x)  # Pass input through network to get the mask
        output = x - torch.flatten(mask)      # Subtract the mask from the input
        return output

# Instantiate model
model = NeuralNetwork().to(device)
print(model)