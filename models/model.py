import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")

"""
Model : 
Conv1 kernel : 21*21 padding: 10*10 512*512
Conv2 kernel : 5*5 padding : 2*2  512*512
Down sample:
Conv3 kernel : 3*3 padding : 1*1 256*256
Conv4 kernel : 3*3 padding : 1*1 256*256
Up Sample:
Conv5 : 1 filter 512*512
Conv1-4 32 filters
"""
"""
TODO: Make this better:

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Convolutional Layer 1: Kernel 21x21, 32 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=21, stride=1, padding=10)
        
        # Convolutional Layer 2: Kernel 5x5, 32 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        # Down-sampling (Conv3 and Conv4): Kernel 3x3, 32 filters, down-sample image size
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)  # Image size 512x512 -> 256x256
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Up-sampling: Conv5, 1 filter to get back to 512x512
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)  # Upsampling to 512x512
    
    def forward(self, x):
        x = self.conv1(x)  # (1, 512, 512) -> (32, 512, 512)
        x = torch.relu(x)
        
        x = self.conv2(x)  # (32, 512, 512) -> (32, 512, 512)
        x = torch.relu(x)
        
        x = self.conv3(x)  # (32, 512, 512) -> (32, 256, 256)
        x = torch.relu(x)
        
        x = self.conv4(x)  # (32, 256, 256) -> (32, 256, 256)
        x = torch.relu(x)
        
        x = self.conv5(x)  # (32, 256, 256) -> (1, 256, 256)
        
        x = self.upsample(x)  # Upsample to 512x512
        
        return x
        """
