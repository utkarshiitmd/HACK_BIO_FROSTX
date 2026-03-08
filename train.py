"""
train.py
--------
Full training loop with:
  - Mixed-precision (AMP)
  - Cosine annealing LR scheduler with warm-up
  - Label smoothing + Focal Loss (handles class imbalance)
  - 5-Fold Cross-Validation
  - Early stopping
  - Automatic checkpoint saving
  - CSV logging (model_performance_analysis.csv)
"""

import os
import csv
import time
import yaml
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedKFold

from data_loader import get_dataloaders, FractureDataset, get_transforms, make_weighted_sampler
from model import build_model
from torch.utils.data import DataLoader, Subset

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# ─── Focal Loss (better than CE for imbalanced classes) ─────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, num_classes=7):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ─── Helper ──────────────────────────────────────────────────────────────────
def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_acc  += accuracy(out.detach(), labels)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item()
        total_acc  += accuracy(out, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


# ─── CSV logger ──────────────────────────────────────────────────────────────
def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "train_accuracy", "val_accuracy",
                         "overfitting_gap", "learning_rate"])
    return path


def log_csv(path, epoch, tr_loss, vl_loss, tr_acc, vl_acc, lr):
    gap = vl_loss - tr_loss
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{tr_loss:.4f}", f"{vl_loss:.4f}",
                         f"{tr_acc:.4f}", f"{vl_acc:.4f}",
                         f"{gap:.4f}", f"{lr:.6f}"])


def append_summary(path, summary: dict):
    with open(path, "a", newline="") as f:
        f.write("\nGENERALIZATION METRICS:\n")
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")


# ─── Main training ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a specific CV fold (0-4); default = full training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Full training run (no CV) ────────────────────────────────────────
    train_loader, val_loader, test_loader, classes = get_dataloaders(cfg)
    num_classes = len(classes)

    model = build_model(cfg, num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model params: {num_params:,}")

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1, num_classes=num_classes)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = GradScaler()

    epochs    = cfg["training"]["epochs"]
    patience  = cfg["training"]["patience"]
    save_dir  = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    no_improve   = 0

    csv_path = save_dir / "model_performance_analysis.csv"
    init_csv(csv_path)

    best_val_epoch = 0
    max_gap = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        gap = abs(vl_loss - tr_loss)
        max_gap = max(max_gap, gap)

        log_csv(csv_path, epoch, tr_loss, vl_loss, tr_acc, vl_acc, lr)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"TrLoss {tr_loss:.4f} TrAcc {tr_acc:.4f} | "
              f"VlLoss {vl_loss:.4f} VlAcc {vl_acc:.4f} | "
              f"LR {lr:.5f} | {time.time()-t0:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_val_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                break

    # ── Final test evaluation ─────────────────────────────────────────────
    model.load_state_dict(torch.load(save_dir / "best_model.pth", map_location=device))
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[Final] Test Accuracy: {te_acc:.4f}")

    append_summary(csv_path, {
        "Max Overfitting Gap": f"{max_gap*100:.2f}%",
        "Best Val Accuracy":   f"{best_val_acc*100:.2f}% (epoch {best_val_epoch})",
        "Test Accuracy":       f"{te_acc*100:.2f}%",
        "Train/Test Accuracy Delta": f"{abs(tr_acc - te_acc)*100:.2f}%",
        "Cross-validation Mean +/- Std": "See cv_results.txt",
    })
    print(f"[Train] Done. CSV saved to {csv_path}")


# ─── 5-Fold CV runner ─────────────────────────────────────────────────────
def run_cross_validation(cfg_path="config.yaml"):
    """
    Call this after main training to get robust CV statistics.
    Saves fold results to checkpoints/cv_results.txt
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = cfg["data"]["root"]

    from data_loader import FractureDataset, get_transforms
    full_ds = FractureDataset(root, "train", get_transforms(cfg, "train"))
    labels  = np.array([s[1] for s in full_ds.samples])
    num_classes = len(full_ds.classes)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n═══ FOLD {fold+1}/5 ═══")
        tr_subset = Subset(full_ds, tr_idx)
        vl_subset = Subset(full_ds, vl_idx)

        tr_loader = DataLoader(tr_subset, batch_size=cfg["training"]["batch_size"],
                               shuffle=True, num_workers=4, pin_memory=True)
        vl_loader = DataLoader(vl_subset, batch_size=cfg["training"]["batch_size"],
                               shuffle=False, num_workers=4, pin_memory=True)

        model     = build_model(cfg, num_classes).to(device)
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                weight_decay=cfg["training"]["weight_decay"])
        scaler    = GradScaler()
        best_acc  = 0.0

        for epoch in range(1, cfg["training"]["cv_epochs"] + 1):
            train_one_epoch(model, tr_loader, optimizer, criterion, scaler, device)
            _, vl_acc = evaluate(model, vl_loader, criterion, device)
            if vl_acc > best_acc:
                best_acc = vl_acc

        fold_accs.append(best_acc)
        print(f"Fold {fold+1} best val acc: {best_acc:.4f}")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    result_str = f"5-Fold CV: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%"
    print(f"\n[CV] {result_str}")

    save_dir = Path(cfg["training"]["save_dir"])
    with open(save_dir / "cv_results.txt", "w") as f:
        f.write(result_str + "\n")
        for i, acc in enumerate(fold_accs):
            f.write(f"Fold {i+1}: {acc*100:.2f}%\n")

    return mean_acc, std_acc


if __name__ == "__main__":
    main()
