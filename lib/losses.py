"""Loss functions for binary segmentation. All take LOGITS (raw model output, no sigmoid)
and a target mask of 0/1 floats with the same shape, e.g. [B, 1, H, W].

Replaces the exercise's lib/losses.py:
- BCELoss:         plain binary cross-entropy (baseline)
- WeightedBCELoss: BCE where foreground pixels count pos_weight times more (class imbalance)
- FocalLoss:       BCE scaled down for pixels that are already easy (Lin et al. 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Plain binary cross-entropy on logits."""
    def forward(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)


class WeightedBCELoss(nn.Module):
    """BCE where positive (foreground) pixels count pos_weight times more.
    Use pos_weight = (#background pixels / #foreground pixels) on the training set."""
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    def forward(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    """Focal loss: BCE scaled by (1 - p_t)^gamma, so confidently-correct pixels count less.
    gamma=2, alpha=0.25 are the defaults from the focal loss paper (and torchvision)."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        p_t = torch.exp(-bce)  # model's probability for the true class of each pixel
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


if __name__ == "__main__":
    # Sanity check: our focal loss should match torchvision's implementation.
    from torchvision.ops import sigmoid_focal_loss
    p = torch.randn(2, 1, 8, 8)
    y = (torch.rand(2, 1, 8, 8) > 0.5).float()
    print("BCE          :", BCELoss()(p, y).item())
    print("Weighted BCE :", WeightedBCELoss(5.0)(p, y).item())
    print("Focal (ours) :", FocalLoss()(p, y).item())
    print("Focal (tv)   :", sigmoid_focal_loss(p, y, reduction="mean").item())