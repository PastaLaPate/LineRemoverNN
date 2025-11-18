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
    return 0.9 * F.l1_loss(pred_logits, target) + 0.1 * (1 - ssim(pred_logits, target, data_range=1.0, size_average=True))