"""
models.py — Three CNN architectures of increasing complexity for FER2013.

Architecture 1 — ShallowCNN   (~600 K params, no BatchNorm)
Architecture 2 — MediumCNN    (~1.5-2 M params, BN + Dropout)
Architecture 3 — SmallResNet  (~2-4 M params, residual blocks + GAP)

All expect input: (batch, 1, 48, 48) float32 tensors, normalised.
All output logits of shape (batch, 7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# Architecture 1 — Shallow CNN (reference / floor model)
# ════════════════════════════════════════════════════════════════════════════
class ShallowCNN(nn.Module):
    """
    Two conv layers, no BatchNorm.
    Purpose: establish the floor. Easy to intentionally overfit or underfit
    by tweaking dropout and learning rate — required by the rubric.

    Expected val accuracy: ~55-60 % at best.
    """

    def __init__(self, dropout: float = 0.3, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1×48×48 → 16×24×24
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 48 → 24

            # Block 2: 16×24×24 → 32×12×12
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 24 → 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # 32*12*12 = 4608
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ════════════════════════════════════════════════════════════════════════════
# Architecture 2 — Medium CNN with BatchNorm + Dropout
# ════════════════════════════════════════════════════════════════════════════
class MediumCNN(nn.Module):
    """
    Three convolutional blocks, each with double conv + BN + Dropout.
    Purpose: show that regularisation closes the overfitting gap from Arch 1.
    Primary hyperparameter sweep target.

    Expected val accuracy: ~60-65 %.
    """

    def __init__(
        self,
        conv_dropout: float = 0.25,
        fc_dropout: float = 0.5,
        num_classes: int = 7,
    ):
        super().__init__()

        def conv_block(in_ch, out_ch, dropout):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            )

        self.features = nn.Sequential(
            conv_block(1,  32,  conv_dropout),   # 48→24, out: 32×24×24
            conv_block(32, 64,  conv_dropout),   # 24→12, out: 64×12×12
            conv_block(64, 128, conv_dropout),   # 12→6,  out: 128×6×6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # 128*6*6 = 4608
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ════════════════════════════════════════════════════════════════════════════
# Architecture 3 — Small Residual CNN
# ════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """Standard pre-activation residual block: Conv→BN→ReLU→Conv→BN + skip."""

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

        # Downsample skip connection if spatial dims change
        self.shortcut = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ChannelTransition(nn.Module):
    """1×1 conv to change channel count between residual groups."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return F.relu(self.proj(x))


class SmallResNet(nn.Module):
    """
    Custom small ResNet designed for 48×48 grayscale input.
    Deliberately smaller than ResNet-18 (which overfits this dataset badly).

    Structure:
        Stem (1→64, 3×3, stride 1) → 2× ResBlock(64) →
        Transition(64→128, stride 2) → 2× ResBlock(128) →
        Transition(128→256, stride 2) → 2× ResBlock(256) →
        GlobalAveragePool → FC(256→7)

    ~2-4 M parameters.
    Expected val accuracy: ~65-68 %.
    """

    def __init__(self, dropout: float = 0.3, num_classes: int = 7):
        super().__init__()

        # Stem: 1×48×48 → 64×48×48
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Group 1: 64×48×48
        self.group1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Transition → Group 2: 128×24×24
        self.trans1  = ChannelTransition(64, 128, stride=2)
        self.group2  = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Transition → Group 3: 256×12×12
        self.trans2  = ChannelTransition(128, 256, stride=2)
        self.group3  = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Head
        self.gap     = nn.AdaptiveAvgPool2d(1)   # 256×1×1
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.trans1(x)
        x = self.group2(x)
        x = self.trans2(x)
        x = self.group3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ════════════════════════════════════════════════════════════════════════════
# Factory helper
# ════════════════════════════════════════════════════════════════════════════
def get_model(arch: str, **kwargs) -> nn.Module:
    """
    arch: "shallow" | "medium" | "resnet"
    kwargs are passed to the model constructor (e.g. dropout=0.4)
    """
    arch = arch.lower()
    if arch == "shallow":
        return ShallowCNN(**kwargs)
    elif arch == "medium":
        return MediumCNN(**kwargs)
    elif arch == "resnet":
        return SmallResNet(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: shallow, medium, resnet")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
