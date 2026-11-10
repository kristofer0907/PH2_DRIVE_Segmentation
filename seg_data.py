"""Data loaders for PH2 (skin lesions) and DRIVE (retinal vessels).

Usage on the cluster:
    python seg_data.py          # builds splits.json and prints a sanity check
"""
import glob, json, os, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

ROOT = "/dtu/datasets1/02516"
PH2_DIR = os.path.join(ROOT, "PH2_Dataset_images")
DRIVE_DIR = os.path.join(ROOT, "DRIVE", "training")  # test/ has no vessel labels


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 file for {pattern}, found {len(hits)}: {hits[:3]}")
    return hits[0]


def list_ph2():
    """Returns [(image_path, mask_path), ...] for every PH2 case."""
    pairs = []
    for case in sorted(os.listdir(PH2_DIR)):
        d = os.path.join(PH2_DIR, case)
        if not os.path.isdir(d):
            continue
        img = find_one(os.path.join(d, "*Dermoscopic_Image*", "*"))
        msk = find_one(os.path.join(d, "*lesion*", "*"))
        pairs.append((img, msk))
    return pairs


def list_drive():
    """Returns [(image_path, vessel_mask_path, fov_mask_path), ...] for labelled DRIVE images."""
    triples = []
    for img in sorted(glob.glob(os.path.join(DRIVE_DIR, "images", "*"))):
        idx = os.path.basename(img).split("_")[0]          # e.g. "21" from "21_training.tif"
        vessel = find_one(os.path.join(DRIVE_DIR, "1st_manual", f"{idx}_*"))  # TARGET: vessel annotation
        fov = find_one(os.path.join(DRIVE_DIR, "mask", f"{idx}_*"))           # field-of-view disc, metrics only
        assert "1st_manual" in vessel, f"vessel mask must come from 1st_manual/, got {vessel}"
        triples.append((img, vessel, fov))
    return triples


def make_splits(n, fractions, seed=42):
    """Shuffle indices 0..n-1 once with a fixed seed and cut into train/val/test."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = round(fractions[0] * n)
    n_va = round(fractions[1] * n)
    return {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}


class SegDataset(Dataset):
    """items: list of tuples (image_path, mask_path[, fov_path]).
    Returns image [3,H,W] float in [0,1], mask [1,H,W] float 0/1, fov [1,H,W] float 0/1 (all ones for PH2)."""

    def __init__(self, items, size, augment=False):
        self.items, self.size, self.augment = items, size, augment

    def __len__(self):
        return len(self.items)

    def _load_mask(self, path):
        m = Image.open(path).convert("L").resize((self.size, self.size), Image.NEAREST)
        return torch.from_numpy((np.array(m) > 127).astype(np.float32))[None]

    def __getitem__(self, i):
        item = self.items[i]
        img = Image.open(item[0]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        img = TF.to_tensor(img)
        mask = self._load_mask(item[1])
        fov = self._load_mask(item[2]) if len(item) > 2 else torch.ones_like(mask)

        if self.augment:  # same flip for image, mask and fov
            if random.random() < 0.5:
                img, mask, fov = TF.hflip(img), TF.hflip(mask), TF.hflip(fov)
            if random.random() < 0.5:
                img, mask, fov = TF.vflip(img), TF.vflip(mask), TF.vflip(fov)
        return img, mask, fov


def get_datasets(name, splits_file="splits.json"):
    """name: 'ph2' or 'drive'. Returns dict of train/val/test datasets using saved splits."""
    with open(splits_file) as f:
        splits = json.load(f)[name]
    items = list_ph2() if name == "ph2" else list_drive()
    size = 256 if name == "ph2" else 512
    return {s: SegDataset([items[i] for i in splits[s]], size, augment=(s == "train"))
            for s in ("train", "val", "test")}


if __name__ == "__main__":
    ph2, drive = list_ph2(), list_drive()
    print(f"PH2: {len(ph2)} cases, e.g. {ph2[0]}")
    print(f"DRIVE: {len(drive)} labelled images, e.g.")
    print(f"   image : {drive[0][0]}\n   target: {drive[0][1]}\n   fov   : {drive[0][2]}")
    for img_p, ves_p, _ in drive:  # a vessel mask covers ~10% of pixels, a FOV disc ~70%
        frac = (np.array(Image.open(ves_p).convert("L")) > 127).mean()
        if frac > 0.3:
            raise ValueError(f"{ves_p} covers {frac:.0%} of the image: that is a disc, not vessels")
    print("   all DRIVE targets look like vessel masks (<30% foreground)")

    splits = {"ph2": make_splits(len(ph2), (0.70, 0.15)),
              "drive": make_splits(len(drive), (0.70, 0.15))}
    with open("splits.json", "w") as f:
        json.dump(splits, f, indent=1)
    for k, v in splits.items():
        print(k, {s: len(ix) for s, ix in v.items()})

    for name in ("ph2", "drive"):
        img, mask, fov = get_datasets(name)["train"][0]
        print(f"{name}: img {tuple(img.shape)}, mask {tuple(mask.shape)}, "
              f"foreground {mask.mean():.3f}, fov {fov.mean():.3f}")
