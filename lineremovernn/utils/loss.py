import torch
import torch.nn.functional as F


def ssim_loss(predictions, targets, kernel_size=11, C1=0.01**2, C2=0.03**2):
    # Dynamic padding based on kernel size to preserve image dimensions
    padding = kernel_size // 2

    # Compute mean for predictions and targets
    mu_x = F.avg_pool2d(predictions, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(targets, kernel_size, stride=1, padding=padding)

    # Compute variance & covariance (clamped to 0 to prevent float32 precision dipping below zero)
    sigma_x = torch.clamp(
        F.avg_pool2d(predictions**2, kernel_size, stride=1, padding=padding) - mu_x**2,
        min=0,
    )
    sigma_y = torch.clamp(
        F.avg_pool2d(targets**2, kernel_size, stride=1, padding=padding) - mu_y**2,
        min=0,
    )
    sigma_xy = (
        F.avg_pool2d(predictions * targets, kernel_size, stride=1, padding=padding)
        - mu_x * mu_y
    )

    # Compute SSIM score
    ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_score = ssim_numerator / ssim_denominator

    return 1 - ssim_score.mean()


def criterion(predicted, target) -> torch.Tensor:
    return F.l1_loss(predicted, target) * 1.0 + ssim_loss(predicted, target) * 0.3
