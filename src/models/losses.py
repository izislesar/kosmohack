"""Segmentation losses for extreme foreground/background imbalance (AF: 0.035% fire).

Dice + Focal (default) or Dice + BCE (config switch). All losses accept an
optional per-pixel `valid` weight (AF valid band): invalid pixels contribute 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if valid is not None:
            v = valid.float()
            probs = probs * v
            targets = targets.float() * v
        else:
            targets = targets.float()
        dims = tuple(range(1, probs.ndim))
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims) + self.smooth
        dice = (2 * inter + self.smooth) / denom
        return (1 - dice).mean()


class FocalLossBinary(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * focal
        if valid is not None:
            v = valid.float()
            return (focal * v).sum() / v.sum().clamp_min(1.0)
        return focal.mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0, alpha=0.8, gamma=2.0, smooth=1.0):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.focal = FocalLossBinary(alpha, gamma)
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits, targets, valid=None):
        return self.dw * self.dice(logits, targets, valid) + self.fw * self.focal(logits, targets, valid)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, pos_weight=50.0, smooth=1.0):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.pos_weight = torch.tensor([pos_weight])
        self.bw = bce_weight
        self.dw = dice_weight

    def forward(self, logits, targets, valid=None):
        pw = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw, reduction="none")
        if valid is not None:
            v = valid.float()
            bce = (bce * v).sum() / v.sum().clamp_min(1.0)
        else:
            bce = bce.mean()
        return self.dw * self.dice(logits, targets, valid) + self.bw * bce


def build_loss(cfg: dict) -> nn.Module:
    kind = cfg.get("kind", "dice+focal")
    if kind == "dice+focal":
        return DiceFocalLoss(
            dice_weight=cfg.get("dice_weight", 1.0),
            focal_weight=cfg.get("focal_weight", 1.0),
            alpha=cfg.get("focal_alpha", 0.8),
            gamma=cfg.get("focal_gamma", 2.0),
            smooth=cfg.get("dice_smooth", 1.0),
        )
    if kind == "dice+bce":
        return DiceBCELoss(
            dice_weight=cfg.get("dice_weight", 1.0),
            bce_weight=1.0,
            pos_weight=cfg.get("bce_pos_weight", 50.0),
            smooth=cfg.get("dice_smooth", 1.0),
        )
    raise ValueError(f"unknown loss kind: {kind}")
