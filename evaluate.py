"""
evaluate.py
-----------
Runs full evaluation on the held-out test set and generates:
  1. final_results.csv  — all required metrics
  2. Confusion matrix heatmap PNG
  3. Per-class GradCAM visualizations

Usage:
    python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import csv
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

from data_loader import get_dataloaders, load_config
from model import build_model, GradCAM


# ─── Inference ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with autocast():
            logits = model(imgs)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_probs.append(probs)
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


# ─── Save final_results.csv ──────────────────────────────────────────────────
def save_results_csv(preds, labels, probs, classes, out_path):
    acc      = accuracy_score(labels, preds)
    per_prec = precision_score(labels, preds, average=None, zero_division=0)
    per_rec  = recall_score(labels, preds, average=None, zero_division=0)
    per_f1   = f1_score(labels, preds, average=None, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # AUC-ROC (one-vs-rest, macro)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(labels, preds)

    rows = []
    header = ["metric_name", "overall_value"] + \
             [f"class_{c}" for c in classes] + ["interpretation"]

    rows.append(["Accuracy",   f"{acc:.4f}"] + ["N/A"] * len(classes) + ["Overall correctness"])
    rows.append(["Precision",  f"{np.mean(per_prec):.4f}"] +
                [f"{v:.4f}" for v in per_prec] + ["Positive prediction reliability"])
    rows.append(["Recall",     f"{np.mean(per_rec):.4f}"] +
                [f"{v:.4f}" for v in per_rec] + ["Detection rate per fracture type"])
    rows.append(["F1-Score",   f"{macro_f1:.4f}"] +
                [f"{v:.4f}" for v in per_f1] + ["Balanced performance per class"])
    rows.append(["AUC-ROC",    f"{auc:.4f}"] + ["N/A"] * len(classes) + ["Threshold-independent discriminability"])
    rows.append(["Macro_F1",   f"{macro_f1:.4f}"] + ["N/A"] * len(classes) + ["Primary ranking metric"])

    # Confusion matrix rows
    rows.append([])
    rows.append(["CONFUSION_MATRIX"] + classes + [""])
    for i, row in enumerate(cm):
        rows.append([f"True_{classes[i]}"] + list(map(str, row)) + [""])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[Evaluate] Accuracy : {acc:.4f}")
    print(f"[Evaluate] Macro F1  : {macro_f1:.4f}")
    print(f"[Evaluate] AUC-ROC   : {auc:.4f}")
    print(f"[Evaluate] CSV saved → {out_path}")
    return acc, macro_f1, auc, cm


# ─── Confusion matrix plot ───────────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*1.4), max(6, len(classes)*1.2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Bone Fracture Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Confusion matrix → {out_path}")


# ─── GradCAM visualizations ──────────────────────────────────────────────────
def save_gradcam_samples(model, test_loader, classes, device, out_dir, n_samples=3):
    """Save GradCAM overlays for a few test samples per class."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model)

    import torchvision.transforms.functional as TF
    from PIL import Image

    seen = {c: 0 for c in range(len(classes))}
    for imgs, labels in test_loader:
        for img, lbl in zip(imgs, labels):
            cls_idx = lbl.item()
            if seen[cls_idx] >= n_samples:
                continue
            heatmap = cam(img.to(device), cls_idx)

            # Resize heatmap to image size
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.ptp() + 1e-8)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Original X-Ray")
            axes[0].axis("off")
            axes[1].imshow(img_np, cmap="gray")
            axes[1].imshow(heatmap, cmap="jet", alpha=0.4)
            axes[1].set_title(f"GradCAM: {classes[cls_idx]}")
            axes[1].axis("off")
            plt.tight_layout()

            fname = out_dir / f"gradcam_{classes[cls_idx]}_{seen[cls_idx]}.png"
            plt.savefig(fname, dpi=120)
            plt.close()
            seen[cls_idx] += 1

        if all(v >= n_samples for v in seen.values()):
            break
    print(f"[Evaluate] GradCAM images saved → {out_dir}/")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--out_dir",    default="outputs")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, classes = get_dataloaders(cfg)
    num_classes = len(classes)

    model = build_model(cfg, num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"[Evaluate] Loaded checkpoint: {args.checkpoint}")

    preds, labels, probs = run_inference(model, test_loader, device)

    acc, macro_f1, auc, cm = save_results_csv(
        preds, labels, probs, classes, out / "final_results.csv"
    )
    plot_confusion_matrix(cm, classes, out / "confusion_matrix.png")
    save_gradcam_samples(model, test_loader, classes, device, out / "gradcam")


if __name__ == "__main__":
    main()
