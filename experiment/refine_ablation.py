"""
refine_ablation.py — Ablation runner for the FineAMT refine pipeline.

Two ablation modes are supported via ``--ablation``:

* ``no_midi``    : the beat-MIDI label tokens are *never* spliced into the
                   input sequence (label-token rows stay zero). The model
                   sees only the spectrogram tokens; the correction head
                   receives no supervision. This isolates the contribution
                   of conditioning on (perturbed) labels at the input.
* ``no_perturb`` : the clean beat-MIDI labels are spliced into the input
                   sequence verbatim, with **no** bit flips applied
                   (p_row = 0, p_flip = 0). The correction head receives no
                   supervision (no row is ever flagged corrupt). This
                   isolates the effect of label perturbation while keeping
                   the label-conditioning input pathway intact.

Both modes share the rest of the refine training/validation pipeline
(model, optimizer, scheduler, metrics, viz, checkpointing) defined in
``refine_experiment.py``.

Usage:
    python experiment/refine_ablation.py \
        --ablation no_midi \
        --dataset_dir    dataset/corpus/MAESTRO-V3/dataset \
        --checkpoint_dir checkpoints/ablation_no_midi \
        --wandb_project  refine-amt-ablation \
        --epochs 20
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.multiprocessing as _torch_mp
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    _torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

import wandb
from dataset.refine_dataset import RefineDataset
from dataset.perturb import perturb_labels
from models.fine import FineAMT
from components.schedulers import make_optimizer, make_scheduler

from experiment.refine_experiment import (
    _KEYS,
    _accumulate_prf,
    _compute_note_counts,
    _extract_beat_preds,
    _finalize_prf,
    _flat_targets,
    _move_batch,
    _new_counts,
    _splice_perturbed_into_sequence,
    _step_loss,
    collate_refine,
    log_checkpoint_artifact,
    log_visualizations,
    save_checkpoint,
)


ABLATIONS = ("no_midi", "no_perturb")


# ============================================================================
# Train / validate (ablation-aware)
# ============================================================================

def train_one_epoch_ablation(
    model:             FineAMT,
    loader:            DataLoader,
    optimizer:         torch.optim.Optimizer,
    device:            torch.device,
    ablation:          str,
    scheduler:         Optional[object] = None,
    scaler:            Optional[torch.amp.GradScaler] = None,
    threshold:         float = 0.5,
    p_row:             float = 0.5,
    p_flip:            float = 0.03,
    lambda_correction: float = 1.0,
    pos_weight_onset:  float = 30.0,
    pos_weight_frame:  float = 5.0,
    pos_weight_offset: float = 40.0,
    focal_gamma_onset: float = 2.0,
    focal_gamma_offset: float = 2.0,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.train()
    total_loss = 0.0
    n_samples  = 0

    fine_counts = _new_counts()
    corr_counts = _new_counts()

    use_amp = scaler is not None

    for batch in tqdm(loader, desc=f"train[{ablation}]", leave=False):
        batch = _move_batch(batch, device)
        type_ids    = batch["type_ids"]
        midi_mask   = batch["midi_mask"]
        midi_labels = batch["midi_labels"]

        if ablation == "no_midi":
            # Skip splicing: label rows stay zero. No perturbation is applied
            # and the correction head receives no supervision.
            correction_mask = torch.zeros_like(midi_mask, dtype=torch.bool)
            seq_in = batch["sequence"]
        elif ablation == "no_perturb":
            # Splice clean labels (no flips). correction_mask is all-False so
            # the correction loss is identically zero.
            seq_in = _splice_perturbed_into_sequence(
                batch["sequence"], type_ids, midi_mask, midi_labels,
            )
            correction_mask = torch.zeros_like(midi_mask, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown ablation: {ablation!r}")

        model_batch = {"sequence": seq_in, "type_ids": type_ids}
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                model_out = model(model_batch)
                loss, fine_preds, corr_preds, targets = _step_loss(
                    model_out, type_ids, midi_mask, midi_labels,
                    correction_mask, lambda_correction,
                    pos_weight_onset, pos_weight_frame, pos_weight_offset,
                    focal_gamma_onset, focal_gamma_offset,
                )
        else:
            model_out = model(model_batch)
            loss, fine_preds, corr_preds, targets = _step_loss(
                model_out, type_ids, midi_mask, midi_labels,
                correction_mask, lambda_correction,
                pos_weight_onset, pos_weight_frame, pos_weight_offset,
                focal_gamma_onset, focal_gamma_offset,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        n = fine_preds["on"].shape[0]
        total_loss += loss.item() * n
        n_samples  += n

        _accumulate_prf(fine_counts, fine_preds, targets, threshold)

    avg_loss = total_loss / max(1, n_samples)
    elem_metrics: Dict[str, float] = {}
    elem_metrics.update(_finalize_prf(fine_counts, prefix="fine_"))
    elem_metrics.update(_finalize_prf(corr_counts, prefix="corr_"))
    note_metrics = {"note_precision": 0.0, "note_recall": 0.0, "note_f1": 0.0}
    return avg_loss, elem_metrics, note_metrics


@torch.no_grad()
def validate_one_epoch_ablation(
    model:             FineAMT,
    loader:            DataLoader,
    device:            torch.device,
    ablation:          str,
    threshold:         float = 0.5,
    note_eval_frac:    float = 0.1,
    p_row:             float = 0.5,
    p_flip:            float = 0.03,
    lambda_correction: float = 1.0,
    pos_weight_onset:  float = 30.0,
    pos_weight_frame:  float = 5.0,
    pos_weight_offset: float = 40.0,
    focal_gamma_onset: float = 2.0,
    focal_gamma_offset: float = 2.0,
    sweep_min:         float = 0.03,
    sweep_max:         float = 0.5,
    sweep_steps:       int = 16,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_samples  = 0

    fine_counts = _new_counts()
    corr_counts = _new_counts()
    note_tp = note_fp = note_fn = 0

    thresholds = torch.linspace(sweep_min, sweep_max, max(2, sweep_steps), dtype=torch.float32)
    sweep_counts = {
        k: {
            "tp": torch.zeros_like(thresholds, dtype=torch.float64),
            "fp": torch.zeros_like(thresholds, dtype=torch.float64),
            "fn": torch.zeros_like(thresholds, dtype=torch.float64),
        }
        for k in _KEYS
    }

    for batch in tqdm(loader, desc=f"val[{ablation}]", leave=False):
        batch = _move_batch(batch, device)
        type_ids    = batch["type_ids"]
        midi_mask   = batch["midi_mask"]
        midi_labels = batch["midi_labels"]

        if ablation == "no_midi":
            correction_mask = torch.zeros_like(midi_mask, dtype=torch.bool)
            seq_in = batch["sequence"]
        elif ablation == "no_perturb":
            seq_in = _splice_perturbed_into_sequence(
                batch["sequence"], type_ids, midi_mask, midi_labels,
            )
            correction_mask = torch.zeros_like(midi_mask, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown ablation: {ablation!r}")

        model_out = model({"sequence": seq_in, "type_ids": type_ids})
        loss, fine_preds, corr_preds, targets = _step_loss(
            model_out, type_ids, midi_mask, midi_labels,
            correction_mask, lambda_correction,
            pos_weight_onset, pos_weight_frame, pos_weight_offset,
            focal_gamma_onset, focal_gamma_offset,
        )

        n = fine_preds["on"].shape[0]
        total_loss += loss.item() * n
        n_samples  += n

        _accumulate_prf(fine_counts, fine_preds, targets, threshold)

        for k in _KEYS:
            pred_k = fine_preds[k].detach().cpu()
            tgt_k = (targets[k].detach().cpu() > 0.5)
            pred_bin = pred_k.unsqueeze(0) > thresholds.view(-1, 1, 1)
            tgt_exp = tgt_k.unsqueeze(0)
            sweep_counts[k]["tp"] += (pred_bin & tgt_exp).sum(dim=(1, 2)).to(torch.float64)
            sweep_counts[k]["fp"] += (pred_bin & ~tgt_exp).sum(dim=(1, 2)).to(torch.float64)
            sweep_counts[k]["fn"] += ((~pred_bin) & tgt_exp).sum(dim=(1, 2)).to(torch.float64)

        if np.random.random() < note_eval_frac:
            fine_on  = model_out["fine"]["on"]
            fine_off = model_out["fine"]["off"]
            B = type_ids.shape[0]
            for i in range(B):
                ti = type_ids[i] == 1
                mi = midi_mask[i]
                if not ti.any() or not mi.any():
                    continue
                n_tp, n_fp, n_fn = _compute_note_counts(
                    fine_on[i][ti].cpu(),                 fine_off[i][ti].cpu(),
                    midi_labels["on"][i][mi].cpu(),       midi_labels["off"][i][mi].cpu(),
                    threshold=threshold,
                )
                note_tp += n_tp; note_fp += n_fp; note_fn += n_fn

    avg_loss = total_loss / max(1, n_samples)
    elem_metrics: Dict[str, float] = {}
    elem_metrics.update(_finalize_prf(fine_counts, prefix="fine_"))
    elem_metrics.update(_finalize_prf(corr_counts, prefix="corr_"))

    macro_f1 = torch.zeros_like(thresholds, dtype=torch.float64)
    for k in _KEYS:
        tp = sweep_counts[k]["tp"]
        fp = sweep_counts[k]["fp"]
        fn = sweep_counts[k]["fn"]
        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)
        f1 = 2.0 * p * r / (p + r + 1e-7)
        best_idx = int(torch.argmax(f1).item())
        elem_metrics[f"sweep_best_{k}_threshold"] = float(thresholds[best_idx].item())
        elem_metrics[f"sweep_best_{k}_f1"] = float(f1[best_idx].item())
        macro_f1 += f1
    macro_f1 = macro_f1 / len(_KEYS)
    best_macro_idx = int(torch.argmax(macro_f1).item())
    elem_metrics["sweep_best_macro_threshold"] = float(thresholds[best_macro_idx].item())
    elem_metrics["sweep_best_macro_f1"] = float(macro_f1[best_macro_idx].item())

    prec = note_tp / (note_tp + note_fp + 1e-7)
    rec  = note_tp / (note_tp + note_fn + 1e-7)
    note_metrics = {
        "note_precision": prec,
        "note_recall":    rec,
        "note_f1":        2.0 * prec * rec / (prec + rec + 1e-7),
    }
    return avg_loss, elem_metrics, note_metrics


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FineAMT ablation runner: no_midi or no_perturb."
    )
    parser.add_argument("--ablation", required=True, choices=ABLATIONS,
                        help="Which ablation to run.")
    # Paths
    parser.add_argument("--dataset_dir",    required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints/ablation")
    # Dataset
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--dt",     type=float, default=256/16000)
    # Model
    parser.add_argument("--dim",      type=int, default=384)
    parser.add_argument("--d_state",  type=int, default=128)
    parser.add_argument("--d_conv",   type=int, default=4)
    parser.add_argument("--expand",   type=int, default=2)
    parser.add_argument("--max_len",  type=int, default=4096)
    parser.add_argument("--n_heads",  type=int, default=8)
    parser.add_argument("--n_experts",type=int, default=8)
    parser.add_argument("--top_k",    type=int, default=2)
    # Training
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--optimizer",  default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler",  default="cosine",
                        choices=["onecycle", "cosine", "constant", "linear", "exponential"])
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true",
                        help="Keep DataLoader workers alive across epochs. "
                             "Off by default: with the file_system sharing "
                             "strategy and two loaders (train+val) live at "
                             "once, FD count climbs and workers SIGABRT.")
    parser.add_argument("--amp",        action="store_true")
    # Loss weights kept identical to refine_experiment so head losses are comparable.
    parser.add_argument("--lambda_correction", type=float, default=1.0)
    parser.add_argument("--pos_weight_onset",  type=float, default=30.0)
    parser.add_argument("--pos_weight_frame",  type=float, default=5.0)
    parser.add_argument("--pos_weight_offset", type=float, default=40.0)
    parser.add_argument("--focal_gamma_onset",  type=float, default=2.0)
    parser.add_argument("--focal_gamma_offset", type=float, default=2.0)
    parser.add_argument("--threshold_sweep_min",   type=float, default=0.03)
    parser.add_argument("--threshold_sweep_max",   type=float, default=0.5)
    parser.add_argument("--threshold_sweep_steps", type=int,   default=16)
    # Wandb
    parser.add_argument("--wandb_project", default="refine-amt-ablation")
    parser.add_argument("--wandb_name",    default=None)
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_log_checkpoints", action="store_true")
    parser.add_argument("--wandb_ckpt_alias", default="latest")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    # Force perturbation off in both ablations so the configuration is
    # self-consistent if these values get logged or replayed.
    args.p_row  = 0.0
    args.p_flip = 0.0

    if args.wandb_name is None:
        args.wandb_name = f"ablation-{args.ablation}"

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Ablation: {args.ablation}")

    # ── Wandb ────────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    if is_main:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   config=vars(args), tags=[args.ablation, "ablation"])
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = RefineDataset(
        split_dir=os.path.join(args.dataset_dir, "train"),
        n_mels=args.n_mels, dt=args.dt,
    )
    val_ds = RefineDataset(
        split_dir=os.path.join(args.dataset_dir, "valid"),
        n_mels=args.n_mels, dt=args.dt,
    )
    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_refine,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ────────────────────────────────────────────────────────────────
    model = FineAMT(
        dim=args.dim, n_mels=args.n_mels,
        d_state=args.d_state, d_conv=args.d_conv, expand=args.expand,
        max_len=args.max_len, n_heads=args.n_heads,
        n_experts=args.n_experts, top_k=args.top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FineAMT: {n_params:,} trainable parameters")
    wandb.config.update({"n_params": n_params})

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    optimizer = make_optimizer(model, optimizer_type=args.optimizer,
                               lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, scheduler_type=args.scheduler,
                               epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                               max_lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}: {args.resume}")

    config_snapshot = vars(args)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} [{args.ablation}] ===")

        tr_loss, tr_metrics, tr_note = train_one_epoch_ablation(
            model, train_loader, optimizer, device,
            ablation=args.ablation,
            scheduler=scheduler, scaler=scaler,
            threshold=args.threshold,
            p_row=args.p_row, p_flip=args.p_flip,
            lambda_correction=args.lambda_correction,
            pos_weight_onset=args.pos_weight_onset,
            pos_weight_frame=args.pos_weight_frame,
            pos_weight_offset=args.pos_weight_offset,
            focal_gamma_onset=args.focal_gamma_onset,
            focal_gamma_offset=args.focal_gamma_offset,
        )

        va_loss, va_metrics, va_note = validate_one_epoch_ablation(
            model, val_loader, device,
            ablation=args.ablation,
            threshold=args.threshold,
            p_row=args.p_row, p_flip=args.p_flip,
            lambda_correction=args.lambda_correction,
            pos_weight_onset=args.pos_weight_onset,
            pos_weight_frame=args.pos_weight_frame,
            pos_weight_offset=args.pos_weight_offset,
            focal_gamma_onset=args.focal_gamma_onset,
            focal_gamma_offset=args.focal_gamma_offset,
            sweep_min=args.threshold_sweep_min,
            sweep_max=args.threshold_sweep_max,
            sweep_steps=args.threshold_sweep_steps,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  loss  train={tr_loss:.4f}  val={va_loss:.4f}  lr={current_lr:.2e}\n"
            f"  fine onset  train_f1={tr_metrics['fine_on_f1']:.4f}  val_f1={va_metrics['fine_on_f1']:.4f}\n"
            f"  fine frame  train_f1={tr_metrics['fine_frame_f1']:.4f}  val_f1={va_metrics['fine_frame_f1']:.4f}\n"
            f"  sweep best threshold={va_metrics['sweep_best_macro_threshold']:.3f}  macro_f1={va_metrics['sweep_best_macro_f1']:.4f}\n"
            f"  note         val_f1={va_note['note_f1']:.4f}"
        )

        log_dict: Dict[str, float] = {"epoch": epoch, "lr": current_lr,
                                       "train/loss": tr_loss, "val/loss": va_loss}
        for k, v in tr_metrics.items(): log_dict[f"train/{k}"] = v
        for k, v in va_metrics.items(): log_dict[f"val/{k}"]   = v
        for k, v in tr_note.items():    log_dict[f"train/{k}"] = v
        for k, v in va_note.items():    log_dict[f"val/{k}"]   = v
        wandb.log(log_dict, step=epoch)

        all_metrics = {**{f"train_{k}": v for k, v in tr_metrics.items()},
                       **{f"val_{k}":   v for k, v in va_metrics.items()},
                       "train_loss": tr_loss, "val_loss": va_loss,
                       "ablation": args.ablation}
        ckpt_path, milestone_path = save_checkpoint(
            model, optimizer, scheduler, epoch,
            all_metrics, args.checkpoint_dir, config_snapshot,
        )

        if is_main and args.wandb_log_checkpoints and not args.wandb_offline:
            log_checkpoint_artifact(
                checkpoint_path=ckpt_path, milestone_path=milestone_path,
                epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                alias=args.wandb_ckpt_alias,
            )

        if len(val_ds) > 0:
            log_visualizations(model, val_ds, epoch, device, args.threshold, np_rng)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wandb.finish()
    print(f"\nAblation '{args.ablation}' complete.")


if __name__ == "__main__":
    main()
