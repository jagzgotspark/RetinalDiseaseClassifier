"""
Day 1: Data loading, preprocessing, and EfficientNet-B0 setup
APTOS 2019 Blindness Detection — 5-class diabetic retinopathy grading

Dataset: https://www.kaggle.com/competitions/aptos2019-blindness-detection
Download train.csv and train_images/ folder from Kaggle, place in ./data/
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
IMG_DIR = os.path.join(DATA_DIR, "train_images")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Label map ────────────────────────────────────────────────────────────────
GRADE_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


# ── Dataset ──────────────────────────────────────────────────────────────────
class APTOSDataset(Dataset):
    """APTOS 2019 retinal fundus image dataset."""

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id_code"] + ".png")
        image = Image.open(img_path).convert("RGB")
        label = int(row["diagnosis"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Preprocessing ─────────────────────────────────────────────────────────────
def get_transforms(train=True):
    """Return data augmentation transforms for train/val splits."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


# ── Weighted sampler to handle class imbalance ────────────────────────────────
def make_weighted_sampler(labels):
    """
    APTOS is heavily imbalanced (most samples are grade 0).
    WeightedRandomSampler gives rarer classes equal representation per batch.
    """
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ── Model: EfficientNet-B0 fine-tuned ─────────────────────────────────────────
def build_model(num_classes=5, freeze_backbone=False):
    """
    Load EfficientNet-B0 pretrained on ImageNet.
    Replace the classifier head for 5-class DR grading.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier: 1280 → num_classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Data split & loader setup ─────────────────────────────────────────────────
def prepare_dataloaders(csv_path, img_dir, val_split=0.15):
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    for grade, name in GRADE_LABELS.items():
        n = (df["diagnosis"] == grade).sum()
        print(f"  Grade {grade} ({name}): {n} ({100*n/len(df):.1f}%)")

    # Stratified split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["diagnosis"], random_state=SEED
    )
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

    train_ds = APTOSDataset(train_df, img_dir, transform=get_transforms(train=True))
    val_ds = APTOSDataset(val_df, img_dir, transform=get_transforms(train=False))

    sampler = make_weighted_sampler(train_df["diagnosis"].tolist())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader, train_df, val_df


# ── Quick sanity check ────────────────────────────────────────────────────────
def visualize_samples(df, img_dir, n=5):
    """Plot a few sample fundus images with their grade labels."""
    samples = df.groupby("diagnosis").first().reset_index()
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
    for ax, (_, row) in zip(axes, samples.iterrows()):
        img_path = os.path.join(img_dir, row["id_code"] + ".png")
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        ax.imshow(img)
        grade = int(row["diagnosis"])
        ax.set_title(f"Grade {grade}\n{GRADE_LABELS[grade]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.show()
    print("Saved: sample_images.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, train_df, val_df = prepare_dataloaders(CSV_PATH, IMG_DIR)
    visualize_samples(train_df, IMG_DIR)

    model = build_model(num_classes=5)
    model = model.to(device)

    # Count trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientNet-B0")
    print(f"Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
    print("\nDay 1 complete. Run day2_train.py next.")