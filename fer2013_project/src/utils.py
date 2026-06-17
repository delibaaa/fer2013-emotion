"""
utils.py — Evaluation helpers: accuracy, macro-F1, confusion matrix, plots.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on Colab)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Tuple, Dict
import wandb

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# ── Evaluation loop ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Returns (avg_loss, accuracy, macro_f1).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item()
        n_batches  += 1
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / n_batches
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ── Confusion matrix ─────────────────────────────────────────────────────────
@torch.no_grad()
def get_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return confusion_matrix(all_labels, all_preds)


def plot_confusion_matrix(cm: np.ndarray, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm / cm.sum(axis=1, keepdims=True),   # normalise rows
        annot=True, fmt=".2f", cmap="Blues",
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    return fig


# ── Training curves ──────────────────────────────────────────────────────────
def plot_training_curves(
    train_accs: list, val_accs: list,
    train_losses: list, val_losses: list,
    title: str = "",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_accs) + 1)

    axes[0].plot(epochs, train_losses, label="train loss")
    axes[0].plot(epochs, val_losses,   label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="train acc")
    axes[1].plot(epochs, val_accs,   label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title or "Training Curves", fontsize=13)
    plt.tight_layout()
    return fig


# ── Per-class report ─────────────────────────────────────────────────────────
@torch.no_grad()
def print_classification_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0))


# ── Wandb logging helpers ────────────────────────────────────────────────────
def log_confusion_matrix_to_wandb(cm: np.ndarray, title: str = "confusion_matrix"):
    fig = plot_confusion_matrix(cm, title=title)
    wandb.log({title: wandb.Image(fig)})
    plt.close(fig)


def log_curves_to_wandb(
    train_accs, val_accs, train_losses, val_losses, title=""
):
    fig = plot_training_curves(train_accs, val_accs, train_losses, val_losses, title)
    wandb.log({"training_curves": wandb.Image(fig)})
    plt.close(fig)
