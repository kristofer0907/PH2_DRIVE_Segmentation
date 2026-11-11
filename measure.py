"""Segmentation metrics: Dice, IoU, Accuracy, Sensitivity, Specificity.

All metrics are computed per image and then averaged over the images,
so a small lesion counts as much as a large one.

Run `python metrics.py` to check the metrics on a tiny mask with known answers.
"""
import torch

METRIC_NAMES = ["dice", "iou", "accuracy", "sensitivity", "specificity"]


def batch_metrics(logits, target, valid=None, threshold=0.5, eps=1e-7):
    """logits, target, valid: tensors of shape [B, 1, H, W].
    valid (optional): 1 where pixels should be counted (e.g. the DRIVE field of view).
    Returns a dict of per-image metric tensors, each of shape [B]."""
    pred = (torch.sigmoid(logits) > threshold).float()
    if valid is None:
        valid = torch.ones_like(target)
    dims = (1, 2, 3)  # sum over channel, height, width -> one count per image
    tp = (pred * target * valid).sum(dims)
    fp = (pred * (1 - target) * valid).sum(dims)
    fn = ((1 - pred) * target * valid).sum(dims)
    tn = ((1 - pred) * (1 - target) * valid).sum(dims)
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
    }


if __name__ == "__main__":
    # Toy check on a 4x4 image (16 pixels):
    # target lesion = top-left 2x2 block (4 pixels)
    # prediction    = top row, left 3 pixels (3 pixels)
    # -> TP = 2, FP = 1, FN = 2, TN = 11
    target = torch.zeros(1, 1, 4, 4); target[0, 0, :2, :2] = 1
    pred = torch.zeros(1, 1, 4, 4); pred[0, 0, 0, :3] = 1
    logits = pred * 20 - 10  # large positive where predicted lesion, large negative elsewhere
    got = {k: round(v.item(), 4) for k, v in batch_metrics(logits, target).items()}
    expected = {"dice": round(4 / 7, 4), "iou": round(2 / 5, 4), "accuracy": round(13 / 16, 4),
                "sensitivity": 0.5, "specificity": round(11 / 12, 4)}
    for k in METRIC_NAMES:
        status = "OK " if abs(got[k] - expected[k]) < 1e-3 else "BAD"
        print(f"{status} {k:12s} got {got[k]:.4f}  expected {expected[k]:.4f}")
