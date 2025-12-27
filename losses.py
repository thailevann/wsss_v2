"""
Loss functions for BCSS-WSSS training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for frequency domain supervision"""
    
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.alpha = float(alpha)

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        pred_freq = torch.fft.fft2(pred, norm="ortho")
        target_freq = torch.fft.fft2(target, norm="ortho")

        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)

        diff = pred_freq - target_freq
        loss = diff.pow(2).sum(dim=-1)

        target_amp = torch.abs(torch.fft.fft2(target, norm="ortho"))
        pred_amp = torch.abs(torch.fft.fft2(pred, norm="ortho"))
        weight = (pred_amp - target_amp).pow(2)
        weight = weight ** self.alpha

        return (loss * weight).mean() * self.loss_weight


def gan_bce_logits(pred, target_is_real: bool):
    """Binary cross-entropy for GAN training"""
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)

