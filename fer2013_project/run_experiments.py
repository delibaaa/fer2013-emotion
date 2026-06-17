"""
run_experiments.py — Launch all experiments in sequence.

This script runs every combination needed for the assignment:
  - Architecture 1 (shallow):  best + overfit run + underfit run
  - Architecture 2 (medium):   best + hyperparameter sweep (LR, dropout, WD)
  - Architecture 3 (resnet):   best + label smoothing variants

Run from repo root:
    python run_experiments.py --csv data/train.csv

On Colab, use the notebook instead (it handles wandb login interactively).
"""

import subprocess
import sys
import argparse


def run(cmd: str):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ⚠️  Command exited with code {result.returncode}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           type=str, default="data/train.csv")
    p.add_argument("--wandb_project", type=str, default="fer2013-emotion")
    p.add_argument("--wandb_entity",  type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    entity_flag = f"--wandb_entity {args.wandb_entity}" if args.wandb_entity else ""
    base = f"python src/train.py --csv {args.csv} --wandb_project {args.wandb_project} {entity_flag}"

    # ── ARCHITECTURE 1: ShallowCNN ──────────────────────────────────────────
    # 1a. Best config for this architecture
    run(f"{base} --arch shallow --epochs 40 --lr 1e-3 --dropout 0.3 --batch_size 64 --run_name shallow_best")

    # 1b. Intentional OVERFIT: no dropout, high LR, no weight decay
    run(f"{base} --arch shallow --epochs 40 --lr 5e-3 --dropout 0.0 --weight_decay 0.0 --batch_size 64 --run_name shallow_overfit")

    # 1c. Intentional UNDERFIT: too much dropout, tiny LR
    run(f"{base} --arch shallow --epochs 40 --lr 1e-4 --dropout 0.7 --batch_size 64 --run_name shallow_underfit")

    # ── ARCHITECTURE 2: MediumCNN ───────────────────────────────────────────
    # 2a. Best config
    run(f"{base} --arch medium --epochs 60 --lr 3e-4 --dropout 0.25 --weight_decay 1e-4 --batch_size 64 --run_name medium_best")

    # 2b. Higher LR (tends to overfit / unstable)
    run(f"{base} --arch medium --epochs 60 --lr 1e-3 --dropout 0.25 --weight_decay 0.0 --batch_size 64 --run_name medium_highlr")

    # 2c. Lower LR, more dropout (underfit territory)
    run(f"{base} --arch medium --epochs 60 --lr 1e-4 --dropout 0.5 --weight_decay 1e-3 --batch_size 64 --run_name medium_lowlr_highdrop")

    # 2d. Large batch (often underfits with same LR)
    run(f"{base} --arch medium --epochs 60 --lr 3e-4 --dropout 0.25 --batch_size 256 --run_name medium_largebatch")

    # ── ARCHITECTURE 3: SmallResNet ─────────────────────────────────────────
    # 3a. Best config
    run(f"{base} --arch resnet --epochs 80 --lr 1e-3 --dropout 0.3 --label_smoothing 0.1 --weight_decay 1e-4 --batch_size 64 --run_name resnet_best")

    # 3b. No label smoothing (often slightly overfit on noisy FER2013 labels)
    run(f"{base} --arch resnet --epochs 80 --lr 1e-3 --dropout 0.3 --label_smoothing 0.0 --batch_size 64 --run_name resnet_no_smoothing")

    # 3c. High weight decay
    run(f"{base} --arch resnet --epochs 80 --lr 1e-3 --dropout 0.2 --weight_decay 1e-3 --batch_size 64 --run_name resnet_high_wd")

    print("\n\n✅ All experiments finished. Check your wandb dashboard.")


if __name__ == "__main__":
    main()
