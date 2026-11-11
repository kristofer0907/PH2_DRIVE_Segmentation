"""Train and evaluate a segmentation model on PH2 or DRIVE.

Examples:
    python train.py --dataset ph2   --model encdec --loss bce   --epochs 30
    python train.py --dataset drive --model unet   --loss focal --epochs 50 --batch_size 2

What it does:
1. trains on the train split, validates after every epoch,
2. keeps the checkpoint with the best validation Dice,
3. evaluates that checkpoint on train/val/test and appends the 5 metrics to results.csv,
4. writes the loss curves to history/<run_name>.csv (for figure F3).
"""
import argparse, csv, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from seg_data import get_datasets
from measure import batch_metrics, METRIC_NAMES
from lib.losses import BCELoss, WeightedBCELoss, FocalLoss
from lib.model.EncDecModel import EncDec


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ph2", "drive"], required=True)
    p.add_argument("--model", choices=["encdec", "unet"], default="encdec")
    p.add_argument("--loss", choices=["bce", "wbce", "focal"], default="bce")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_fov", action="store_true", help="DRIVE: count all pixels in metrics, not only the field of view")
    p.add_argument("--tag", default="", help="optional extra text in the run name")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(name):
    if name == "encdec":
        return EncDec()
    from lib.model.UNetModel import UNet  # imported here so EncDec runs work before the U-Net exists
    return UNet()


def foreground_ratio(dataset):
    """pos_weight = #background / #foreground pixels over the training set (no augmentation effect on counts)."""
    fg = total = 0.0
    for _, mask, _ in dataset:
        fg += mask.sum().item(); total += mask.numel()
    return (total - fg) / fg


def build_loss(name, train_set, device):
    if name == "bce":
        return BCELoss()
    if name == "focal":
        return FocalLoss()
    pw = foreground_ratio(train_set)
    print(f"pos_weight (background/foreground) = {pw:.2f}")
    return WeightedBCELoss(pw).to(device)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_fov):
    """Returns (mean loss, dict of mean metrics) over all images in the loader."""
    model.eval()
    total_loss, n_batches, per_image = 0.0, 0, {k: [] for k in METRIC_NAMES}
    for X, Y, fov in loader:
        X, Y, fov = X.to(device), Y.to(device), fov.to(device)
        logits = model(X)
        total_loss += loss_fn(logits, Y).item(); n_batches += 1
        m = batch_metrics(logits, Y, fov if use_fov else None)
        for k in METRIC_NAMES:
            per_image[k] += m[k].cpu().tolist()
    return total_loss / max(n_batches, 1), {k: float(np.mean(v)) for k, v in per_image.items()}


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = f"{args.model}_{args.dataset}_{args.loss}" + (f"_{args.tag}" if args.tag else "")
    use_fov = args.dataset == "drive" and not args.no_fov
    print(f"Run {run} on {device}")

    ds = get_datasets(args.dataset)
    loaders = {
        "train": DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=3),
        "train_eval": DataLoader(get_datasets(args.dataset)["train"], batch_size=args.batch_size, num_workers=3),
        "val": DataLoader(ds["val"], batch_size=args.batch_size, num_workers=3),
        "test": DataLoader(ds["test"], batch_size=args.batch_size, num_workers=3),
    }
    loaders["train_eval"].dataset.augment = False  # evaluate training images without flips
    print(f"Images: {len(ds['train'])} train / {len(ds['val'])} val / {len(ds['test'])} test")

    model = build_model(args.model).to(device)
    loss_fn = build_loss(args.loss, ds["train"], device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True); os.makedirs("history", exist_ok=True)
    ckpt = f"checkpoints/{run}.pt"
    best_dice, best_val_loss, history = -1.0, float("inf"), []

    for epoch in range(1, args.epochs + 1):
        tic = time.time()
        model.train()
        train_loss = 0.0
        for X, Y, _ in loaders["train"]:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            opt.step()
            train_loss += loss.item() / len(loaders["train"])

        val_loss, val_m = evaluate(model, loaders["val"], loss_fn, device, use_fov)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_m["dice"]})
        flag = ""
        # best = highest val Dice; on a tie (e.g. Dice stuck at 0) prefer the lower val loss
        if val_m["dice"] > best_dice or (val_m["dice"] == best_dice and val_loss < best_val_loss):
            best_dice, best_val_loss = val_m["dice"], val_loss
            torch.save(model.state_dict(), ckpt)
            flag = "  <- best, saved"
        print(f"Epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val dice {val_m['dice']:.4f}  ({time.time() - tic:.0f}s){flag}")

    with open(f"history/{run}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)

    # Final evaluation of the best checkpoint on all three splits
    model.load_state_dict(torch.load(ckpt, map_location=device))
    new_file = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["run", "dataset", "model", "loss", "split"] + METRIC_NAMES)
        for split, key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
            _, m = evaluate(model, loaders[key], loss_fn, device, use_fov)
            w.writerow([run, args.dataset, args.model, args.loss, split] + [f"{m[k]:.4f}" for k in METRIC_NAMES])
            print(f"{split:5s} " + "  ".join(f"{k} {m[k]:.4f}" for k in METRIC_NAMES))
    print(f"Done. Best val Dice {best_dice:.4f}. Results appended to results.csv")


if __name__ == "__main__":
    main()