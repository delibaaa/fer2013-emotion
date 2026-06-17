"""
data.py — FER2013 dataset loading, splitting, and augmentation.

FER2013 format: CSV with columns [emotion, pixels, Usage]
  - emotion: integer 0-6
  - pixels: space-separated string of 48*48=2304 pixel values (0-255)
  - Usage: "Training", "PublicTest", "PrivateTest"

We ignore the built-in Usage split and do our own stratified 90/10 train/val
split so every experiment uses the same held-out validation set.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ── Label mapping ────────────────────────────────────────────────────────────
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
NUM_CLASSES = 7


# ── Dataset class ────────────────────────────────────────────────────────────
class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for FER2013.

    Args:
        df         : pandas DataFrame with columns [emotion, pixels]
        transform  : torchvision transform pipeline (or None)
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.labels = df["emotion"].values.astype(np.int64)
        # Parse pixel strings → (N, 48, 48) float32 arrays normalised to [0,1]
        self.images = np.array(
            [np.fromstring(p, sep=" ", dtype=np.float32).reshape(48, 48) / 255.0
             for p in df["pixels"]]
        )
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Shape: (1, 48, 48) — single grayscale channel
        img = torch.tensor(self.images[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label


# ── Transform pipelines ──────────────────────────────────────────────────────
def get_transforms(augment: bool = True):
    """
    augment=True  → training transforms (random flips, crops, colour jitter)
    augment=False → validation/test transforms (just normalise)

    Mean/std computed on FER2013 training pixels: roughly mean≈0.508, std≈0.255.
    Using these instead of ImageNet values since images are greyscale faces.
    """
    mean = [0.508]
    std  = [0.255]

    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(size=48, scale=(0.85, 1.0)),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=mean, std=std),
        ])


# ── Data loading helper ──────────────────────────────────────────────────────
def load_fer2013(csv_path: str, val_size: float = 0.1, seed: int = 42):
    """
    Load the FER2013 train.csv, stratified-split into train/val,
    and return DataFrames plus class weights for loss reweighting.

    Returns:
        train_df, val_df, class_weights (torch.FloatTensor of shape [7])
    """
    df = pd.read_csv(csv_path)

    # Keep only Training rows from the original split
    # (PrivateTest/PublicTest have no labels usable for validation here)
    train_df = df[df["Usage"] == "Training"].copy().reset_index(drop=True)

    # Stratified 90/10 split
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["emotion"],
        random_state=seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    # Class weights: inverse-frequency weighting, useful for 'disgust' (class 1)
    counts = np.bincount(train_df["emotion"].values, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()  # normalise so they sum to 1
    class_weights = torch.FloatTensor(weights)

    print(f"[data] train={len(train_df)}  val={len(val_df)}")
    print(f"[data] class counts (train): {counts.astype(int).tolist()}")
    print(f"[data] class weights: {class_weights.tolist()}")

    return train_df, val_df, class_weights


def get_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so every mini-batch has roughly equal
    class representation — especially important for the tiny 'disgust' class.
    """
    labels = train_df["emotion"].values
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    sample_weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def get_dataloaders(
    csv_path: str,
    batch_size: int = 64,
    val_size: float = 0.1,
    use_weighted_sampler: bool = True,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    End-to-end helper: CSV → (train_loader, val_loader, class_weights).

    Args:
        csv_path            : path to FER2013 train.csv
        batch_size          : mini-batch size
        val_size            : fraction of training rows to hold out for val
        use_weighted_sampler: oversample minority classes during training
        num_workers         : DataLoader workers (0 on Colab if issues arise)
        seed                : reproducibility seed
    """
    train_df, val_df, class_weights = load_fer2013(csv_path, val_size, seed)

    train_transform = get_transforms(augment=True)
    val_transform   = get_transforms(augment=False)

    train_dataset = FER2013Dataset(train_df, transform=train_transform)
    val_dataset   = FER2013Dataset(val_df,   transform=val_transform)

    sampler = get_weighted_sampler(train_df) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # don't shuffle if sampler is provided
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights
