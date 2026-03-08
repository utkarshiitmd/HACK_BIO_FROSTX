"""
data_loader.py
--------------
Handles all dataset downloading, preprocessing, and DataLoader creation
for the Bone Fracture Multi-Region X-ray Classification task.

Dataset: https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-imaging
"""

import os
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ─── Load config ────────────────────────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# ─── Dataset ────────────────────────────────────────────────────────────────
class FractureDataset(Dataset):
    """
    Loads bone X-ray images from a directory structure like:
        data/
          train/
            class_A/  *.jpg / *.png
            class_B/  ...
          val/
            ...
          test/
            ...
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.samples = []  # list of (path, label_idx)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in (self.root / cls).glob(ext):
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root}. "
                               f"Make sure the Kaggle dataset is unzipped into data/")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            print(f"Bad image skipped: {path}")
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Transforms ─────────────────────────────────────────────────────────────
def get_transforms(cfg, split: str):
    img_size = cfg["data"]["img_size"]
    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─── Weighted sampler (handle class imbalance) ────────────────────────────
def make_weighted_sampler(dataset: FractureDataset):
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─── DataLoader factory ──────────────────────────────────────────────────
def get_dataloaders(cfg):
    root = cfg["data"]["root"]
    batch = cfg["training"]["batch_size"]
    workers = cfg["data"].get("num_workers", 4)

    train_ds = FractureDataset(root, "train", get_transforms(cfg, "train"))
    val_ds   = FractureDataset(root, "val",   get_transforms(cfg, "val"))
    test_ds  = FractureDataset(root, "test",  get_transforms(cfg, "test"))

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                              num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True)

    print(f"[DataLoader] Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.classes