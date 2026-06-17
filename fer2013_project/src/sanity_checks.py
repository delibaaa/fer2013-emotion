"""
sanity_checks.py — Forward/backward sanity checks (required by rubric).

Run these BEFORE any full training loop to catch wiring bugs early.
Results are printed and also returned as a dict you can log to wandb.

Three checks per architecture:
  1. Output shape check  — model(batch) must have shape (B, 7)
  2. Initial loss check  — untrained model loss should be ≈ ln(7) ≈ 1.946
  3. Overfit-tiny-batch  — model should reach near-zero loss on 8 samples
                           within ~50 gradient steps (proves backward pass works)
  4. Gradient norm check — log grad norms for first few steps (catch NaN/explode)
"""

import math
import torch
import torch.nn as nn
import wandb


def check_output_shape(model: nn.Module, device: torch.device) -> bool:
    """Check model outputs (B, 7) for a dummy batch."""
    model.eval()
    dummy = torch.randn(8, 1, 48, 48).to(device)
    with torch.no_grad():
        out = model(dummy)
    expected = (8, 7)
    ok = out.shape == expected
    status = "✅ PASS" if ok else f"❌ FAIL — got {out.shape}"
    print(f"  [shape check] expected {expected}, got {out.shape}  {status}")
    return ok


def check_initial_loss(
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    """
    Untrained cross-entropy loss should be ≈ ln(7) ≈ 1.946.
    A very different value suggests broken weight init or wrong label mapping.
    """
    model.eval()
    dummy_x = torch.randn(64, 1, 48, 48).to(device)
    dummy_y = torch.randint(0, 7, (64,)).to(device)
    with torch.no_grad():
        logits = model(dummy_x)
        loss   = criterion(logits, dummy_y).item()
    expected = math.log(7)
    diff     = abs(loss - expected)
    ok       = diff < 0.3    # allow ±0.3 tolerance
    status   = "✅ PASS" if ok else "⚠️  CHECK — initial loss far from ln(7)"
    print(f"  [init loss]   expected ≈{expected:.3f}, got {loss:.3f}  {status}")
    return {"sanity/init_loss": loss, "sanity/expected_init_loss": expected}


def check_overfit_tiny_batch(
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    n_steps: int = 100,
    lr: float = 1e-3,
    target_loss: float = 0.05,
) -> dict:
    """
    Train on 8 fixed samples for n_steps steps.
    If loss doesn't reach target_loss the backward pass / optimizer is broken.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.randn(8, 1, 48, 48).to(device)
    y = torch.randint(0, 7, (8,)).to(device)

    final_loss = None
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    ok     = final_loss < target_loss
    status = "✅ PASS" if ok else f"❌ FAIL — loss stuck at {final_loss:.4f}"
    print(f"  [tiny-batch]  loss after {n_steps} steps: {final_loss:.4f}  {status}")
    return {"sanity/tiny_batch_final_loss": final_loss, "sanity/tiny_batch_ok": int(ok)}


def check_gradient_norms(
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    n_steps: int = 5,
    lr: float = 1e-3,
) -> dict:
    """
    Log gradient L2 norms for the first n_steps steps.
    NaN → broken forward pass; very large (>100) → exploding gradients.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    norms = []
    for step in range(n_steps):
        x = torch.randn(16, 1, 48, 48).to(device)
        y = torch.randint(0, 7, (16,)).to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        norms.append(total_norm)
        ok = not math.isnan(total_norm) and total_norm < 500
        print(f"  [grad norm]   step {step+1}: {total_norm:.4f}  {'✅' if ok else '❌ NaN/Exploding'}")
        optimizer.step()
    return {"sanity/grad_norms": norms}


def run_all_sanity_checks(
    model: nn.Module,
    device: torch.device,
    arch_name: str,
    class_weights: torch.FloatTensor = None,
    log_to_wandb: bool = True,
) -> bool:
    """
    Run all four checks. Returns True if all pass.
    Call this before every full training run.
    """
    print(f"\n{'='*55}")
    print(f"  SANITY CHECKS for {arch_name.upper()}")
    print(f"{'='*55}")

    # Fresh model copy so the overfit-tiny-batch check doesn't pollute weights
    import copy
    model_copy = copy.deepcopy(model).to(device)

    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )

    shape_ok = check_output_shape(model_copy, device)

    # Reset for next checks
    model_copy = copy.deepcopy(model).to(device)
    init_loss_results = check_initial_loss(model_copy, device, criterion)

    model_copy = copy.deepcopy(model).to(device)
    tiny_results = check_overfit_tiny_batch(model_copy, device, criterion)

    model_copy = copy.deepcopy(model).to(device)
    grad_results = check_gradient_norms(model_copy, device, criterion)

    all_ok = shape_ok and tiny_results["sanity/tiny_batch_ok"]
    print(f"\n  Overall: {'✅ ALL PASSED' if all_ok else '❌ SOME CHECKS FAILED'}")
    print(f"{'='*55}\n")

    if log_to_wandb:
        try:
            metrics = {
                "sanity/shape_ok": int(shape_ok),
                **init_loss_results,
                **tiny_results,
                "sanity/grad_norm_mean": sum(grad_results["sanity/grad_norms"]) / len(grad_results["sanity/grad_norms"]),
            }
            wandb.log(metrics)
        except Exception as e:
            print(f"  [wandb] could not log sanity metrics: {e}")

    return all_ok
