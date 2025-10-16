import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.center = ConvBlock(512, 1024)

        self.att4 = AttentionBlock(512, 512, 256)
        self.att3 = AttentionBlock(256, 256, 128)
        self.att2 = AttentionBlock(128, 128, 64)
        self.att1 = AttentionBlock(64, 64, 32)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec4 = ConvBlock(1024, 512)
        self.dec3 = ConvBlock(512, 256)
        self.dec2 = ConvBlock(256, 128)
        self.dec1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        center = self.center(self.pool(e4))

        d4 = self.up4(center)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)  # raw
"""
class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=21, padding=10, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(32)
        self.lr1 = nn.LeakyReLU(inplace=True)
        self.l1 = nn.Sequential(self.conv1, self.bn1, self.lr1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=11, padding=5, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(32)
        self.lr2 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Sequential(self.conv2, self.bn2, self.lr2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, padding=3, padding_mode='reflect')
        self.bn3 = nn.BatchNorm2d(32)
        self.lr3 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Sequential(self.conv3, self.bn3, self.lr3)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1, padding_mode='reflect')
        self.lr4 = nn.LeakyReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(1)
        self.l4 = nn.Sequential(self.conv4, self.bn4, self.lr4)

    def forward(self, _in):

        x = self.l1(_in)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x - _in