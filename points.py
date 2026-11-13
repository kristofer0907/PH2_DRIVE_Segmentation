"""Part 2: weak supervision with simulated point clicks (PH2).

- sample_clicks(): simulates an annotator clicking N positive points (on the lesion)
  and N negative points (background), using one of two strategies:
    "random": uniform random pixels inside / outside the lesion mask, simplest approach, naive tho.
    "center": people choose what looks the most that they're looking for, which is in the center
    , therefore we decrease the odds of the edges of the items being selected.
- PointDataset: wraps the normal training set; returns (image, click_labels, click_mask)
  where click_mask = 1 only on clicked pixels. Clicks are generated once with a fixed
  seed, so every run with the same settings uses exactly the same clicks.
- PointLoss: partial cross-entropy = BCE averaged over clicked pixels only
  (Bearman et al., ECCV 2016, eq. 3, without the image-level term).

Run `python points.py` to save clicks_preview.png (figure F6) with both strategies.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

STRATEGIES = ("random", "center")


def _pick(weights, n, rng):
    """Pick up to n distinct pixels with probability proportional to weights. 
    Returns [k, 2] (row, col)."""
    flat = weights.ravel()
    idx = np.flatnonzero(flat > 0)
    if len(idx) == 0:
        return np.zeros((0, 2), dtype=int)
    p = flat[idx] / flat[idx].sum()
    chosen = rng.choice(idx, size=min(n, len(idx)), replace=False, p=p)
    return np.stack(np.unravel_index(chosen, weights.shape), axis=1)


def sample_clicks(mask, n_pos, n_neg, strategy, rng):
    """mask: [H, W] array of 0/1. Returns (pos, neg),
      each an array of [k, 2] pixel coordinates."""
    fg = mask > 0.5
    bg = ~fg
    if strategy == "random":
        pos_w, neg_w = fg.astype(float), bg.astype(float)
    elif strategy == "center":
        pos_w = distance_transform_edt(fg) ** 2           # deep inside the lesion = likely click
        dist_out = distance_transform_edt(bg)             # distance from the lesion, for background pixels
        band = bg & (dist_out >= 5) & (dist_out <= 30)    # background just around the lesion
        neg_w = band.astype(float) if band.sum() >= n_neg else bg.astype(float)
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return _pick(pos_w, n_pos, rng), _pick(neg_w, n_neg, rng)


class PointDataset(Dataset):
    """Wraps a SegDataset (from seg_data.get_datasets(...)['train']) with simulated clicks."""

    def __init__(self, base, n_clicks, strategy, seed=0, augment=True):
        self.base, self.augment = base, augment
        self.base.augment = False  # flips are applied here, jointly to image and clicks
        rng = np.random.default_rng(seed)
        self.clicks = []
        for i in range(len(base)):
            _, mask, _ = base[i]
            self.clicks.append(sample_clicks(mask[0].numpy(), n_clicks, n_clicks, strategy, rng))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _, _ = self.base[i]
        H, W = img.shape[1:]
        labels = torch.zeros(1, H, W)
        cmask = torch.zeros(1, H, W)
        pos, neg = self.clicks[i]
        labels[0, pos[:, 0], pos[:, 1]] = 1
        cmask[0, pos[:, 0], pos[:, 1]] = 1
        cmask[0, neg[:, 0], neg[:, 1]] = 1
        if self.augment:
            if random.random() < 0.5:
                img, labels, cmask = TF.hflip(img), TF.hflip(labels), TF.hflip(cmask)
            if random.random() < 0.5:
                img, labels, cmask = TF.vflip(img), TF.vflip(labels), TF.vflip(cmask)
        return img, labels, cmask


class PointLoss(nn.Module):
    """Partial binary cross-entropy: only clicked pixels contribute; all others are ignored."""
    def forward(self, y_pred, y_true, click_mask):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        return (bce * click_mask).sum() / click_mask.sum().clamp(min=1)


if __name__ == "__main__":
    # Figure F6: one PH2 training image with clicks from both strategies.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seg_data import get_datasets

    base = get_datasets("ph2")["train"]
    base.augment = False
    img, mask, _ = base[0]
    rgb, m = img.permute(1, 2, 0).numpy(), mask[0].numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, strategy in zip(ax, STRATEGIES):
        pos, neg = sample_clicks(m, 10, 10, strategy, np.random.default_rng(0))
        a.imshow(rgb)
        a.contour(m, levels=[0.5], colors="white", linewidths=1)
        a.scatter(pos[:, 1], pos[:, 0], c="lime", s=40, edgecolors="black", label="positive")
        a.scatter(neg[:, 1], neg[:, 0], c="red", s=40, edgecolors="black", label="negative")
        a.set_title(f"strategy: {strategy} (10 + 10 clicks)")
        a.axis("off")
    ax[0].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("clicks_preview.png", dpi=100)
    print("saved clicks_preview.png")

    # Sanity check of the loss: ignores unclicked pixels
    loss = PointLoss()
    pred = torch.randn(1, 1, 8, 8)
    lab, cm = torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)
    cm[0, 0, 0, 0] = 1
    a = loss(pred, lab, cm).item()
    pred2 = pred.clone(); pred2[0, 0, 5, 5] += 100  # change an unclicked pixel
    b = loss(pred2, lab, cm).item()
    print("PointLoss ignores unclicked pixels:", abs(a - b) < 1e-6)
