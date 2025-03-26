import torch
import torch.nn as nn
import torchmetrics.functional as TF
from torchvision import models


class SSIM_L1_Loss(nn.Module):
    def __init__(self, alpha=0.8):
        """
        SSIM + L1 Loss for image restoration.
        :param alpha: Weight for SSIM loss. (1 - alpha) will be used for L1 loss.
        """
        super(SSIM_L1_Loss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        """
        Compute the SSIM + L1 loss.
        :param pred: Predicted image tensor (BxCxHxW).
        :param target: Ground truth image tensor (BxCxHxW).
        :return: Weighted SSIM + L1 loss.
        """
        # SSIM Loss (1 - SSIM because we minimize the loss)
        ssim_loss = 1 - TF.structural_similarity_index_measure(
            pred, target, data_range=1.0
        )

        # L1 Loss
        l1_loss = self.l1_loss(pred, target)

        # Combined loss
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


class BCEL1Loss(nn.Module):
    def __init__(self, l1_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        return self.bce(pred, target) + self.l1_weight * self.l1(pred, target)


# VGG + L1 Loss
class VGGLoss(nn.Module):
    def __init__(self, l1_weight=0.1):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[LineRemoverNN] VGG Loss Using", self.device, "device")
        vgg = vgg.to(self.device)
        self.vgg = nn.Sequential(*[layer for layer in vgg])
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)
        vgg_pred = self.vgg(
            pred.repeat(1, 3, 1, 1)
        )  # Convert grayscale to 3-channel for VGG
        vgg_target = self.vgg(target.repeat(1, 3, 1, 1))
        return self.l1_weight * self.l1(vgg_pred, vgg_target) + self.l1(pred, target)
