"""
train.py — Training loop with Wandb logging.

Usage (from repo root):
    python src/train.py --arch shallow --lr 1e-3 --epochs 30 --dropout 0.3
    python src/train.py --arch medium  --lr 3e-4 --epochs 50 --dropout 0.25
    python src/train.py --arch resnet  --lr 1e-3 --epochs 60 --dropout 0.3 --label_smoothing 0.1

Wandb structure mirrors MLflow:
  - Each architecture family = one Wandb "group" (e.g. group="arch1_shallow")
  - Each hyperparameter run   = one Wandb "run"  (auto-named by wandb)
  - Metrics logged per epoch: train_loss, val_loss, train_acc, val_acc, val_f1
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from pathlib import Path

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data         import get_dataloaders
from models       import get_model, count_parameters
from sanity_checks import run_all_sanity_checks
from utils        import (
    evaluate,
    get_confusion_matrix,
    log_confusion_matrix_to_wandb,
    log_curves_to_wandb,
    print_classification_report,
)


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FER2013 training script")
    p.add_argument("--csv",             type=str,   default="data/train.csv",
                   help="Path to FER2013 train.csv")
    p.add_argument("--arch",            type=str,   default="shallow",
                   choices=["shallow", "medium", "resnet"],
                   help="Architecture: shallow | medium | resnet")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing (0 = off). Useful for resnet.")
    p.add_argument("--use_class_weights", action="store_true", default=True,
                   help="Weight loss by inverse class frequency")
    p.add_argument("--use_sampler",     action="store_true", default=True,
                   help="Oversample minority classes with WeightedRandomSampler")
    p.add_argument("--lr_schedule",     type=str,   default="cosine",
                   choices=["none", "cosine", "step"],
                   help="LR scheduler")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--wandb_project",   type=str,   default="fer2013-emotion")
    p.add_argument("--wandb_entity",    type=str,   default=None,
                   help="Your wandb username or team name")
    p.add_argument("--run_name",        type=str,   default=None,
                   help="Optional custom run name")
    p.add_argument("--skip_sanity",     action="store_true",
                   help="Skip sanity checks (not recommended)")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # ── Wandb init ───────────────────────────────────────────────────────────
    # Group = architecture family (mirrors MLflow "experiment")
    # Run   = individual hyperparameter configuration
    run_name = args.run_name or (
        f"{args.arch}_lr{args.lr}_drop{args.dropout}_bs{args.batch_size}"
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=f"arch_{args.arch}",          # one group per architecture family
        name=run_name,
        config=vars(args),
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_weights = get_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        use_weighted_sampler=args.use_sampler,
    )
    wandb.config.update({"n_train": len(train_loader.dataset),
                          "n_val":   len(val_loader.dataset)})

    # ── Model ────────────────────────────────────────────────────────────────
    # Build kwargs for model constructor based on architecture
    model_kwargs = {"dropout": args.dropout}
    if args.arch == "medium":
        model_kwargs = {"conv_dropout": args.dropout, "fc_dropout": 0.5}

    model = get_model(args.arch, **model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"[train] architecture: {args.arch}  |  parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ── Sanity checks ────────────────────────────────────────────────────────
    if not args.skip_sanity:
        cw = class_weights if args.use_class_weights else None
        passed = run_all_sanity_checks(model, device, args.arch, cw, log_to_wandb=True)
        if not passed:
            print("[train] ⚠️  Sanity checks failed — review above before continuing.")

    # ── Loss, optimiser, scheduler ───────────────────────────────────────────
    cw_tensor = class_weights.to(device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(
        weight=cw_tensor,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5
        )
    else:
        scheduler = None

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        running_loss, running_correct, n_samples = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()

            # Gradient clipping for stability (especially useful for resnet)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss    += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            n_samples       += x.size(0)

        train_loss = running_loss / n_samples
        train_acc  = running_correct / n_samples

        if scheduler:
            scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        train_accs.append(train_acc);   val_accs.append(val_acc)
        train_losses.append(train_loss); val_losses.append(val_loss)

        # ── Log to wandb ──────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "train_acc":  train_acc,
            "val_acc":    val_acc,
            "val_f1":     val_f1,
            "lr":         current_lr,
        })

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f} | "
            f"lr={current_lr:.6f}"
        )

        # ── Save best model ───────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{run_name}_best.pt")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n[train] Best val accuracy: {best_val_acc:.4f}")

    # Load best weights for final metrics
    model.load_state_dict(torch.load(f"checkpoints/{run_name}_best.pt", map_location=device))

    print("\n[train] Per-class classification report (val set):")
    print_classification_report(model, val_loader, device)

    cm = get_confusion_matrix(model, val_loader, device)
    log_confusion_matrix_to_wandb(cm, title=f"confusion_matrix_{run_name}")
    log_curves_to_wandb(train_accs, val_accs, train_losses, val_losses, title=run_name)

    # Overfit / underfit diagnosis logged as summary metrics
    train_val_gap = train_accs[-1] - val_accs[-1]
    wandb.summary["best_val_acc"]    = best_val_acc
    wandb.summary["train_val_gap"]   = train_val_gap
    wandb.summary["diagnosis"] = (
        "overfit"   if train_val_gap > 0.12 else
        "underfit"  if val_accs[-1] < 0.45  else
        "good_fit"
    )

    wandb.finish()
    print("\n[train] Done. Run logged to wandb.")


if __name__ == "__main__":
    main()
