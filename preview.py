"""Visual check of the data loaders: image + mask overlay for PH2 and DRIVE.

Run from the project folder (next to seg_data.py and splits.json):
    python preview.py
Then open preview.png (e.g. in VS Code's file explorer).
"""
import matplotlib
matplotlib.use("Agg")  # no display on the cluster: draw to a file
import matplotlib.pyplot as plt
from seg_data import get_datasets

fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for r, name in enumerate(["ph2", "drive"]):
    ds = get_datasets(name)["train"]
    for c in range(2):
        img, mask, fov = ds[c]
        rgb = img.permute(1, 2, 0)  # [3,H,W] -> [H,W,3] for matplotlib
        ax[r, 2 * c].imshow(rgb)
        ax[r, 2 * c].set_title(f"{name} image {c}")
        ax[r, 2 * c + 1].imshow(rgb)
        ax[r, 2 * c + 1].imshow(mask[0], alpha=0.45, cmap="Reds")
        ax[r, 2 * c + 1].set_title(f"{name} mask overlay {c}")
for a in ax.flat:
    a.axis("off")
plt.tight_layout()
plt.savefig("preview.png", dpi=100)
print("saved preview.png")
