import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss，用于缓解前景/背景类别不平衡。"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice 混合损失。

    total_loss = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        if bce_weight <= 0 or dice_weight <= 0:
            raise ValueError("bce_weight 与 dice_weight 必须为正数。")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

    def loss_label(self):
        bce = int(self.bce_weight) if self.bce_weight.is_integer() else self.bce_weight
        dice = int(self.dice_weight) if self.dice_weight.is_integer() else self.dice_weight
        return f"BCE:Dice = {bce}:{dice}"
