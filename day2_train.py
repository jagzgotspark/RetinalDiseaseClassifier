"""
Day 2: Training loop + Grad-CAM explainability
Run: python day2_train.py

Saves:
  best_model.pth       — best checkpoint by quadratic kappa
  gradcam_outputs/     — heatmap visualizations on val images
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns

# Reuse Day 1 helpers
from day1_setup import (
    build_model, prepare_dataloaders, get_transforms,
    APTOSDataset, GRADE_LABELS, IMG_DIR, CSV_PATH, device
)

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_PATH = "best_model.pth"
GRADCAM_DIR = "gradcam_outputs"
os.makedirs(GRADCAM_DIR, exist_ok=True)


# ── Quadratic Weighted Kappa ──────────────────────────────────────────────────
def quadratic_kappa(y_true, y_pred):
    """
    Standard competition metric for APTOS.
    Penalizes predictions that are far from the true grade more heavily.
    Score ranges from -1 (worse than random) to 1.0 (perfect).
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    kappa = quadratic_kappa(all_labels, all_preds)
    return total_loss / len(loader.dataset), kappa, all_preds, all_labels


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["val_kappa"], label="Val kappa", color="green")
    axes[1].set_title("Quadratic Weighted Kappa")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved: training_curves.png")


def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=GRADE_LABELS.values(),
        yticklabels=GRADE_LABELS.values(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — DR Grading")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights the regions in the fundus image the model focuses on
    when predicting each DR grade.

    Works by:
    1. Forward pass → get predicted class
    2. Backprop gradient to the last conv layer
    3. Global-average-pool the gradients → channel weights
    4. Weighted sum of feature maps → heatmap
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Last conv block in EfficientNet-B0 features
        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor, class_idx=None):
        """
        img_tensor: (1, 3, H, W) on device
        Returns: heatmap as numpy array (H, W), values in [0, 1]
        """
        self.model.eval()
        output = self.model(img_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Pool gradients across channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


def overlay_heatmap(img_pil, cam, alpha=0.45):
    """Overlay Grad-CAM heatmap on the original fundus image."""
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    heatmap = cm.jet(cam)[:, :, :3]  # (H, W, 3), drop alpha channel
    overlay = alpha * heatmap + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def run_gradcam_on_val(model, val_df, n_per_class=2):
    """Generate and save Grad-CAM heatmaps for n_per_class samples each grade."""
    gradcam = GradCAM(model)
    transform = get_transforms(train=False)

    for grade in range(5):
        subset = val_df[val_df["diagnosis"] == grade].head(n_per_class)
        for _, row in subset.iterrows():
            img_path = os.path.join(IMG_DIR, row["id_code"] + ".png")
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            cam, pred_class = gradcam.generate(img_tensor)
            overlay = overlay_heatmap(img_pil, cam)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img_pil.resize((224, 224)))
            axes[0].set_title(f"Original\nTrue: Grade {grade} ({GRADE_LABELS[grade]})")
            axes[0].axis("off")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Grad-CAM\nPred: Grade {pred_class} ({GRADE_LABELS[pred_class]})")
            axes[1].axis("off")
            plt.tight_layout()

            fname = f"{GRADCAM_DIR}/grade{grade}_{row['id_code']}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, train_df, val_df = prepare_dataloaders(CSV_PATH, IMG_DIR)
    model = build_model(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_kappa": []}
    best_kappa = -1.0

    print(f"Training for {EPOCHS} epochs on {device}...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_kappa, all_preds, all_labels = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_kappa"].append(val_kappa)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"Kappa: {val_kappa:.4f}"
              + (" ← best" if val_kappa > best_kappa else ""))

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"\nBest quadratic kappa: {best_kappa:.4f}")
    print(f"Checkpoint saved: {CHECKPOINT_PATH}")

    plot_training_curves(history)
    plot_confusion_matrix(all_labels, all_preds)

    # Load best model for Grad-CAM
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("\nGenerating Grad-CAM heatmaps...")
    run_gradcam_on_val(model, val_df, n_per_class=2)
    print("\nDay 2 complete. Run streamlit run day3_app.py next.")