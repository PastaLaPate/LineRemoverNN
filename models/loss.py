from pytorch_msssim import ssim
from torchvision.models import mobilenet_v2
from torch import nn
from torch.nn import functional as F
import torch


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v2(pretrained=True).features[:8]
        for param in m.parameters():
            param.requires_grad = False
        self.m = m.eval()
        self.transform = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, bias=False)  # replicate 1ch -> 3ch
        )

    def forward(self, x, y):
        self.transform = self.transform.to(x.device).to(x.dtype)
        self.m = self.m.to(x.device).to(x.dtype)
        x = self.transform(x)
        y = self.transform(y)
        return F.mse_loss(self.m(x), self.m(y))


def combined_loss(pred_logits, target):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred = torch.sigmoid(pred_logits)  # Only for SSIM and perceptual
    ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    perceptual = PerceptualLoss()(pred, target)
    return bce + 0.5 * ssim_loss + 0.3 * perceptual
