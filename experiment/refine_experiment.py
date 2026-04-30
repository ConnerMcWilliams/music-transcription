"""
refine_experiment.py — Full training + validation pipeline for FineAMT.

Usage:
    python experiment/refine_experiment.py \
        --dataset_dir    dataset/corpus/MAESTRO-V3/built \
        --checkpoint_dir checkpoints/refine \
        --wandb_project  refine-amt \
        --epochs 20

The dataset must have been pre-packed by ``python -m dataset.build_dataset``;
``--dataset_dir`` points at the directory containing ``train/`` and ``valid/``
subfolders of .npy / .npz arrays.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from dataset.refine_dataset import RefineDataset
from dataset.perturb import perturb_labels
from models.fine import FineAMT
from components.schedulers import make_optimizer, make_scheduler


# ============================================================================
# Collate
# ============================================================================

def collate_refine(batch: List[Dict]) -> Dict[str, object]:
    """
    Pad variable-length RefineDataset samples into a fixed-size batch.

    Input (per sample):
        sequence     [T+x, D]
        type_ids     [T+x]         — 0 = spec, 1 = beat
        midi_labels  {k: [x, 128]}

    Output:
        sequence     [B, max_len, D]
        type_ids     [B, max_len]   — -1 = padding
        midi_labels  {k: [B, max_x, 128]}
        midi_mask    [B, max_x]     — True for real beat positions
    """
    sequences    = [s["sequence"]  for s in batch]
    type_ids_raw = [s["type_ids"]  for s in batch]

    # Pad sequences and type_ids
    sequences_pad = pad_sequence(sequences,    batch_first=True, padding_value=0.0)
    type_ids_pad  = pad_sequence(type_ids_raw, batch_first=True, padding_value=-1)

    # Pad midi_labels
    keys   = ("on", "off", "frame")
    x_lens = [s["midi_labels"]["on"].shape[0] for s in batch]
    max_x  = max(x_lens) if x_lens else 0
    B      = len(batch)

    midi_labels_pad: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [s["midi_labels"][k] for s in batch]  # each [x_i, 128]
        midi_labels_pad[k] = pad_sequence(tensors, batch_first=True, padding_value=0.0)

    midi_mask = torch.zeros(B, max_x, dtype=torch.bool)
    for i, x in enumerate(x_lens):
        midi_mask[i, :x] = True

    return {
        "sequence":    sequences_pad,    # [B, max_len, D]
        "type_ids":    type_ids_pad,     # [B, max_len]
        "midi_labels": midi_labels_pad,  # {k: [B, max_x, 128]}
        "midi_mask":   midi_mask,        # [B, max_x]
    }


# ============================================================================
# Metrics
# ============================================================================

def _prf(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> Tuple[float, float, float]:
    """Element-wise precision / recall / F1 from binary tensors."""
    pred = pred_bin.float().flatten()
    gt   = gt_bin.float().flatten()
    tp   = (pred * gt).sum()
    fp   = (pred * (1.0 - gt)).sum()
    fn   = ((1.0 - pred) * gt).sum()
    p    = tp / (tp + fp + 1e-7)
    r    = tp / (tp + fn + 1e-7)
    f1   = 2.0 * p * r / (p + r + 1e-7)
    return p.item(), r.item(), f1.item()


def compute_metrics(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute element-wise precision / recall / F1 for on / off / frame.

    Args:
        preds   : {k: [N, 128]}  — aggregated sigmoid outputs
        targets : {k: [N, 128]}  — aggregated GT
        threshold : binarisation threshold

    Returns:
        Flat dict: {on_precision, on_recall, on_f1, off_*, frame_*}
    """
    metrics: Dict[str, float] = {}
    for k in ("on", "off", "frame"):
        pred_bin = preds[k]   > threshold
        gt_bin   = targets[k] > 0.5
        p, r, f1 = _prf(pred_bin, gt_bin)
        metrics[f"{k}_precision"] = p
        metrics[f"{k}_recall"]    = r
        metrics[f"{k}_f1"]        = f1
    return metrics


def _pair_notes_np(on_bin: np.ndarray, off_bin: np.ndarray) -> np.ndarray:
    """Pair onsets with nearest following offset using searchsorted. Returns [P,2] int32."""
    onsets  = np.where(on_bin)[0]
    offsets = np.where(off_bin)[0]
    if len(onsets) == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(offsets) == 0:
        # No offsets predicted — pair each onset with itself (zero-length note).
        return np.stack([onsets, onsets], axis=1).astype(np.int32)
    idx  = np.searchsorted(offsets, onsets, side="left")
    offs = np.where(idx < len(offsets), offsets[np.minimum(idx, len(offsets) - 1)], onsets)
    return np.stack([onsets, offs], axis=1).astype(np.int32)


def _match_notes_vectorized(pred_notes: np.ndarray, gt_notes: np.ndarray, tolerance: int) -> int:
    """Count matched note TPs using a [P,G] distance matrix."""
    if len(pred_notes) == 0 or len(gt_notes) == 0:
        return 0
    on_diff  = np.abs(pred_notes[:, 0:1] - gt_notes[:, 0])   # [P, G]
    off_diff = np.abs(pred_notes[:, 1:2] - gt_notes[:, 1])   # [P, G]
    match    = (on_diff <= tolerance) & (off_diff <= tolerance)
    matched_gt = np.zeros(len(gt_notes), dtype=bool)
    tp = 0
    for pi in range(len(pred_notes)):
        candidates = np.where(match[pi] & ~matched_gt)[0]
        if len(candidates) > 0:
            matched_gt[candidates[0]] = True
            tp += 1
    return tp


def _compute_note_counts(
    on_pred:   torch.Tensor,
    off_pred:  torch.Tensor,
    on_gt:     torch.Tensor,
    off_gt:    torch.Tensor,
    threshold: float = 0.5,
    tolerance: int   = 2,
) -> Tuple[int, int, int]:
    """Return cumulative (tp, fp, fn) for note-level matching across all pitches."""
    on_p  = (on_pred  > threshold).numpy()
    off_p = (off_pred > threshold).numpy()
    on_g  = (on_gt    > 0.5).numpy()
    off_g = (off_gt   > 0.5).numpy()
    total_tp = total_fp = total_fn = 0
    for pitch in range(on_p.shape[1]):
        pred_notes = _pair_notes_np(on_p[:, pitch], off_p[:, pitch])
        gt_notes   = _pair_notes_np(on_g[:, pitch], off_g[:, pitch])
        tp = _match_notes_vectorized(pred_notes, gt_notes, tolerance)
        total_tp += tp
        total_fp += len(pred_notes) - tp
        total_fn += len(gt_notes)   - tp
    return total_tp, total_fp, total_fn


def compute_note_f1(
    on_pred:  torch.Tensor,
    off_pred: torch.Tensor,
    on_gt:    torch.Tensor,
    off_gt:   torch.Tensor,
    threshold: float = 0.5,
    tolerance: int = 2,
) -> Dict[str, float]:
    """
    Note-level precision / recall / F1 via onset + offset pairing.

    Args:
        on_pred, off_pred : [N, 128] — aggregated sigmoid predictions
        on_gt,  off_gt    : [N, 128] — aggregated GT
        tolerance : frame tolerance for onset/offset matching

    Returns:
        {note_precision, note_recall, note_f1}
    """
    total_tp, total_fp, total_fn = _compute_note_counts(
        on_pred, off_pred, on_gt, off_gt, threshold, tolerance
    )
    prec = total_tp / (total_tp + total_fp + 1e-7)
    rec  = total_tp / (total_tp + total_fn + 1e-7)
    f1   = 2.0 * prec * rec / (prec + rec + 1e-7)
    return {"note_precision": prec, "note_recall": rec, "note_f1": f1}


# ============================================================================
# Training / Validation
# ============================================================================

_KEYS = ("on", "off", "frame")


def _move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    """Move every tensor leaf in a collated batch to ``device``."""
    out: Dict[str, object] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _splice_perturbed_into_sequence(
    sequence:    torch.Tensor,         # [B, max_len, max_dim]
    type_ids:    torch.Tensor,         # [B, max_len]
    midi_mask:   torch.Tensor,         # [B, max_x]
    perturbed:   Dict[str, torch.Tensor],   # {k: [B, max_x, P]}
) -> torch.Tensor:
    """
    Write the perturbed labels into the label-token rows of ``sequence``
    (positions where type_id == 1) **in place**.

    The dataset leaves label rows zero, and ``_move_batch`` returns a fresh
    GPU tensor that is not reused beyond this batch, so in-place mutation is
    safe and avoids a ~100MB clone per step.
    """
    perturbed_concat = torch.cat([perturbed[k] for k in _KEYS], dim=-1)   # [B, max_x, 3P]
    label_dim = perturbed_concat.shape[-1]
    sequence[type_ids == 1, :label_dim] = perturbed_concat[midi_mask]
    return sequence


def _extract_beat_preds(
    head_out:  Dict[str, torch.Tensor],   # {k: [B, max_len, P]}
    type_ids:  torch.Tensor,              # [B, max_len]
) -> Dict[str, torch.Tensor]:
    """Gather head predictions at every type_id==1 position into [N, P]."""
    beat_mask = type_ids == 1
    return {k: head_out[k][beat_mask] for k in _KEYS}


def _flat_targets(
    midi_labels: Dict[str, torch.Tensor],   # {k: [B, max_x, P]}
    midi_mask:   torch.Tensor,              # [B, max_x]
) -> Dict[str, torch.Tensor]:
    return {k: midi_labels[k][midi_mask] for k in _KEYS}


def _step_loss(
    model_out:        Dict[str, Dict[str, torch.Tensor]],
    type_ids:         torch.Tensor,
    midi_mask:        torch.Tensor,
    midi_labels:      Dict[str, torch.Tensor],
    correction_mask:  torch.Tensor,            # [B, max_x]
    lambda_correction: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute fine + correction BCE. Returns (loss, fine_preds, corr_preds, targets)."""
    fine_preds       = _extract_beat_preds(model_out["fine"],       type_ids)
    correction_preds = _extract_beat_preds(model_out["correction"], type_ids)
    targets          = _flat_targets(midi_labels, midi_mask)

    fine_loss = sum(F.binary_cross_entropy(fine_preds[k],       targets[k]) for k in _KEYS)

    corrupt_flat = correction_mask[midi_mask]                      # [N]
    if corrupt_flat.any():
        corr_loss = sum(
            F.binary_cross_entropy(
                correction_preds[k][corrupt_flat],
                targets[k][corrupt_flat],
            )
            for k in _KEYS
        )
    else:
        corr_loss = fine_loss.new_zeros(())

    loss = fine_loss + lambda_correction * corr_loss
    return loss, fine_preds, correction_preds, targets


def _accumulate_prf(
    counts: Dict[str, Dict[str, float]],
    preds:  Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    threshold: float,
) -> None:
    """
    Compute tp/fp/fn on-device for all three keys, then transfer once.

    Six separate ``.cpu()`` transfers per batch was the prior bottleneck on
    the metric path; this issues a single device→host copy of a [3, 3] long
    tensor.
    """
    p = torch.stack([preds[k].detach()  > threshold for k in _KEYS])  # [3, N, P]
    t = torch.stack([target[k].detach() > 0.5       for k in _KEYS])

    tp = ( p &  t).sum(dim=(1, 2))
    fp = ( p & ~t).sum(dim=(1, 2))
    fn = (~p &  t).sum(dim=(1, 2))
    stats = torch.stack([tp, fp, fn]).cpu().tolist()                  # one transfer

    for j, k in enumerate(_KEYS):
        counts["tp"][k] += stats[0][j]
        counts["fp"][k] += stats[1][j]
        counts["fn"][k] += stats[2][j]


def _finalize_prf(counts: Dict[str, Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in _KEYS:
        tp, fp, fn = counts["tp"][k], counts["fp"][k], counts["fn"][k]
        p  = tp / (tp + fp + 1e-7)
        r  = tp / (tp + fn + 1e-7)
        f1 = 2.0 * p * r / (p + r + 1e-7)
        out[f"{prefix}{k}_precision"] = p
        out[f"{prefix}{k}_recall"]    = r
        out[f"{prefix}{k}_f1"]        = f1
    return out


def _new_counts() -> Dict[str, Dict[str, float]]:
    return {kind: {k: 0.0 for k in _KEYS} for kind in ("tp", "fp", "fn")}


def train_one_epoch(
    model:             FineAMT,
    loader:            DataLoader,
    optimizer:         torch.optim.Optimizer,
    device:            torch.device,
    scheduler:         Optional[object] = None,
    scaler:            Optional[torch.amp.GradScaler] = None,
    threshold:         float = 0.5,
    p_row:             float = 0.5,
    p_flip:            float = 0.03,
    lambda_correction: float = 1.0,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    One training epoch with label perturbation.

    A fraction (~p_row) of beat-token rows are eligible for bit flips at rate
    p_flip. The corrupted labels are spliced back into the input sequence so
    the model sees errors; the correction head is supervised on the rows that
    actually got flipped.

    Returns:
        avg_loss     : scalar float
        elem_metrics : {fine/corr}_{on/off/frame}_{precision/recall/f1}
        note_metrics : {note_precision, note_recall, note_f1}  (placeholder)
    """
    model.train()
    total_loss = 0.0
    n_samples  = 0

    fine_counts = _new_counts()
    corr_counts = _new_counts()

    use_amp = scaler is not None

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch(batch, device)
        type_ids    = batch["type_ids"]
        midi_mask   = batch["midi_mask"]
        midi_labels = batch["midi_labels"]

        # Perturb clean labels (CPU/GPU agnostic — uses tensors' device).
        perturbed_labels, correction_mask = perturb_labels(
            midi_labels, midi_mask, p_row=p_row, p_flip=p_flip,
        )

        seq_perturbed = _splice_perturbed_into_sequence(
            batch["sequence"], type_ids, midi_mask, perturbed_labels,
        )
        model_batch = {"sequence": seq_perturbed, "type_ids": type_ids}

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                model_out = model(model_batch)
                loss, fine_preds, corr_preds, targets = _step_loss(
                    model_out, type_ids, midi_mask, midi_labels,
                    correction_mask, lambda_correction,
                )
        else:
            model_out = model(model_batch)
            loss, fine_preds, corr_preds, targets = _step_loss(
                model_out, type_ids, midi_mask, midi_labels,
                correction_mask, lambda_correction,
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

        # Correction metrics measured only on actually-flipped rows.
        corrupt_flat = correction_mask[midi_mask]
        if corrupt_flat.any():
            corr_preds_sub = {k: corr_preds[k][corrupt_flat]   for k in _KEYS}
            corr_target    = {k: targets[k][corrupt_flat]      for k in _KEYS}
            _accumulate_prf(corr_counts, corr_preds_sub, corr_target, threshold)

    avg_loss = total_loss / max(1, n_samples)
    elem_metrics: Dict[str, float] = {}
    elem_metrics.update(_finalize_prf(fine_counts, prefix="fine_"))
    elem_metrics.update(_finalize_prf(corr_counts, prefix="corr_"))
    note_metrics: Dict[str, float] = {
        "note_precision": 0.0,
        "note_recall":    0.0,
        "note_f1":        0.0,
    }
    return avg_loss, elem_metrics, note_metrics


@torch.no_grad()
def validate_one_epoch(
    model:             FineAMT,
    loader:            DataLoader,
    device:            torch.device,
    threshold:         float = 0.5,
    note_eval_frac:    float = 0.1,
    p_row:             float = 0.5,
    p_flip:            float = 0.03,
    lambda_correction: float = 1.0,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    One validation epoch (no gradients), with the same perturbation regime as
    training so the loss is comparable. Reports separate fine_/corr_ metrics
    and a note-level F1 measured on the *fine* head.
    """
    model.eval()
    total_loss = 0.0
    n_samples  = 0

    fine_counts = _new_counts()
    corr_counts = _new_counts()
    note_tp = note_fp = note_fn = 0

    for batch in tqdm(loader, desc="val  ", leave=False):
        batch = _move_batch(batch, device)
        type_ids    = batch["type_ids"]
        midi_mask   = batch["midi_mask"]
        midi_labels = batch["midi_labels"]

        perturbed_labels, correction_mask = perturb_labels(
            midi_labels, midi_mask, p_row=p_row, p_flip=p_flip,
        )
        seq_perturbed = _splice_perturbed_into_sequence(
            batch["sequence"], type_ids, midi_mask, perturbed_labels,
        )
        model_out = model({"sequence": seq_perturbed, "type_ids": type_ids})

        loss, fine_preds, corr_preds, targets = _step_loss(
            model_out, type_ids, midi_mask, midi_labels,
            correction_mask, lambda_correction,
        )

        n = fine_preds["on"].shape[0]
        total_loss += loss.item() * n
        n_samples  += n

        _accumulate_prf(fine_counts, fine_preds, targets, threshold)

        corrupt_flat = correction_mask[midi_mask]
        if corrupt_flat.any():
            corr_preds_sub = {k: corr_preds[k][corrupt_flat] for k in _KEYS}
            corr_target    = {k: targets[k][corrupt_flat]    for k in _KEYS}
            _accumulate_prf(corr_counts, corr_preds_sub, corr_target, threshold)

        if np.random.random() < note_eval_frac:
            n_tp, n_fp, n_fn = _compute_note_counts(
                fine_preds["on"].cpu(),  fine_preds["off"].cpu(),
                targets["on"].cpu(),     targets["off"].cpu(),
                threshold=threshold,
            )
            note_tp += n_tp; note_fp += n_fp; note_fn += n_fn

    avg_loss = total_loss / max(1, n_samples)
    elem_metrics: Dict[str, float] = {}
    elem_metrics.update(_finalize_prf(fine_counts, prefix="fine_"))
    elem_metrics.update(_finalize_prf(corr_counts, prefix="corr_"))

    prec = note_tp / (note_tp + note_fp + 1e-7)
    rec  = note_tp / (note_tp + note_fn + 1e-7)
    note_metrics: Dict[str, float] = {
        "note_precision": prec,
        "note_recall":    rec,
        "note_f1":        2.0 * prec * rec / (prec + rec + 1e-7),
    }
    return avg_loss, elem_metrics, note_metrics


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model:           FineAMT,
    optimizer:       torch.optim.Optimizer,
    scheduler:       Optional[object],
    epoch:           int,
    metrics:         Dict[str, float],
    checkpoint_dir:  str,
    config_snapshot: Dict,
) -> None:
    """Save model checkpoint.  Milestone every 5 epochs."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    payload = {
        "epoch":                 epoch,
        "model_state_dict":      model.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict() if scheduler is not None else None,
        "metrics":               metrics,
        "config":                config_snapshot,
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save(payload, path)

    if epoch % 5 == 0:
        milestone = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}_milestone.pt")
        torch.save(payload, milestone)
        print(f"  Milestone checkpoint: {milestone}")


# ============================================================================
# Visualizations
# ============================================================================

def log_visualizations(
    model:       FineAMT,
    val_dataset: RefineDataset,
    epoch:       int,
    device:      torch.device,
    threshold:   float,
    rng:         np.random.Generator,
) -> None:
    """
    Pick a random val sample, run inference, and log a matplotlib figure to wandb.

    Figure layout (3 rows × 3 columns):
        Row 0 : input spectrogram  (spec tokens, type_id==0)
        Row 1 : GT onset / frame / offset
        Row 2 : predicted onset / frame / offset  (thresholded)
    """
    model.eval()

    idx = int(rng.integers(0, len(val_dataset)))
    sample = val_dataset[idx]

    # Build a single-sample batch
    batch = _move_batch(collate_refine([sample]), device)
    seq      = batch["sequence"]
    type_ids = batch["type_ids"]

    with torch.no_grad():
        model_out = model({"sequence": seq, "type_ids": type_ids})
    fine_out = model_out["fine"]

    # ── spectrogram (spec tokens in chronological order) ──────────────────
    spec_mask = (type_ids[0] == 0)                     # [L]
    spec_tok  = seq[0][spec_mask].cpu().numpy()        # [T, D]
    # Use only the first n_mels dims for display
    n_mels    = min(128, spec_tok.shape[1])
    spec_img  = spec_tok[:, :n_mels].T                 # [n_mels, T]

    # ── GT labels (beat tokens) ────────────────────────────────────────────
    midi_mask = batch["midi_mask"][0]                  # [max_x] (on device)
    gt_on  = batch["midi_labels"]["on"]   [0][midi_mask].cpu().numpy().T   # [128, x]
    gt_off = batch["midi_labels"]["off"]  [0][midi_mask].cpu().numpy().T
    gt_frm = batch["midi_labels"]["frame"][0][midi_mask].cpu().numpy().T

    # ── predictions (beat tokens) ─────────────────────────────────────────
    beat_mask = (type_ids[0] == 1)
    pr_on  = fine_out["on"][0][beat_mask].cpu().numpy().T    # [128, x]
    pr_off = fine_out["off"][0][beat_mask].cpu().numpy().T
    pr_frm = fine_out["frame"][0][beat_mask].cpu().numpy().T
    pr_on_bin  = (pr_on  > threshold).astype(float)
    pr_off_bin = (pr_off > threshold).astype(float)
    pr_frm_bin = (pr_frm > threshold).astype(float)

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 9))
    fig.suptitle(f"Epoch {epoch} — val sample {idx}", fontsize=11)

    cmap_spec  = "magma"
    cmap_label = "Blues"

    # Row 0: spectrogram (span all 3 cols)
    ax = axes[0, 0]
    ax.imshow(spec_img, aspect="auto", origin="lower", cmap=cmap_spec)
    ax.set_title("Input spectrogram")
    ax.set_xlabel("time (frames)")
    ax.set_ylabel("mel bin")
    for a in axes[0, 1:]:
        a.axis("off")

    # Row 1: GT
    for col, (img, title) in enumerate(
        [(gt_on, "GT onset"), (gt_frm, "GT frame"), (gt_off, "GT offset")]
    ):
        axes[1, col].imshow(img, aspect="auto", origin="lower", cmap=cmap_label,
                            vmin=0, vmax=1)
        axes[1, col].set_title(title)
        axes[1, col].set_xlabel("beat subdivisions")
        axes[1, col].set_ylabel("pitch")

    # Row 2: predictions
    for col, (img, title) in enumerate(
        [(pr_on_bin, "Pred onset"), (pr_frm_bin, "Pred frame"), (pr_off_bin, "Pred offset")]
    ):
        axes[2, col].imshow(img, aspect="auto", origin="lower", cmap=cmap_label,
                            vmin=0, vmax=1)
        axes[2, col].set_title(title)
        axes[2, col].set_xlabel("beat subdivisions")
        axes[2, col].set_ylabel("pitch")

    plt.tight_layout()
    wandb.log({"viz/sample": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FineAMT on RefineDataset with wandb logging."
    )
    # Paths
    parser.add_argument("--dataset_dir",    required=True,  help="Pre-built dataset root (contains train/, valid/, test/ subdirs from dataset.build_dataset)")
    parser.add_argument("--checkpoint_dir", default="checkpoints/refine", help="Checkpoint output dir")
    # Dataset
    parser.add_argument("--n_mels",    type=int, default=128,  help="Mel bins in spectrogram")
    parser.add_argument("--dt",        type=float, default=256/16000, help="Seconds per spec frame")
    # Model
    parser.add_argument("--blocks", type=int, default=8,   help="Mamba2 block count")
    parser.add_argument("--dim",    type=int, default=384, help="Model d_model")
    parser.add_argument("--d_state",type=int, default=128)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--max_len",type=int, default=4096)
    # Training
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--optimizer",  default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler",  default="cosine",
                        choices=["onecycle", "cosine", "constant", "linear", "exponential"])
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--num_workers",type=int,
                        default=min(4, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--amp",        action="store_true", help="Use automatic mixed precision")
    # Label perturbation / correction head
    parser.add_argument("--p_row",  type=float, default=0.5,
                        help="Probability that a beat-token row is eligible for label flips")
    parser.add_argument("--p_flip", type=float, default=0.03,
                        help="Per-element flip probability inside an eligible row")
    parser.add_argument("--lambda_correction", type=float, default=1.0,
                        help="Weight on the correction-head BCE loss")
    # Wandb
    parser.add_argument("--wandb_project", default="refine-amt")
    parser.add_argument("--wandb_name",    default=None)
    parser.add_argument("--wandb_offline", action="store_true")
    # Resume
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Wandb ─────────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = RefineDataset(
        split_dir = os.path.join(args.dataset_dir, "train"),
        n_mels    = args.n_mels,
        dt        = args.dt,
    )
    val_ds = RefineDataset(
        split_dir = os.path.join(args.dataset_dir, "valid"),
        n_mels    = args.n_mels,
        dt        = args.dt,
    )
    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    loader_kwargs = dict(
        batch_size  = args.batch_size,
        collate_fn  = collate_refine,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FineAMT(
        blocks  = args.blocks,
        dim     = args.dim,
        n_mels  = args.n_mels,
        d_state = args.d_state,
        d_conv  = args.d_conv,
        expand  = args.expand,
        max_len = args.max_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FineAMT: {n_params:,} trainable parameters")
    wandb.config.update({"n_params": n_params})

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer = make_optimizer(
        model,
        optimizer_type = args.optimizer,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    scheduler = make_scheduler(
        optimizer,
        scheduler_type = args.scheduler,
        epochs         = args.epochs,
        steps_per_epoch= steps_per_epoch,
        max_lr         = args.lr,       # for onecycle
    )

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    # ── Resume ────────────────────────────────────────────────────────────────
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

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # ── Train ────────────────────────────────────────────────────────────
        tr_loss, tr_metrics, tr_note = train_one_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler, scaler=scaler,
            threshold=args.threshold,
            p_row=args.p_row, p_flip=args.p_flip,
            lambda_correction=args.lambda_correction,
        )

        # ── Validate ─────────────────────────────────────────────────────────
        va_loss, va_metrics, va_note = validate_one_epoch(
            model, val_loader, device, threshold=args.threshold,
            p_row=args.p_row, p_flip=args.p_flip,
            lambda_correction=args.lambda_correction,
        )

        # ── LR ───────────────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Print summary ────────────────────────────────────────────────────
        print(
            f"  loss  train={tr_loss:.4f}  val={va_loss:.4f}  lr={current_lr:.2e}\n"
            f"  fine onset  train_f1={tr_metrics['fine_on_f1']:.4f}  val_f1={va_metrics['fine_on_f1']:.4f}\n"
            f"  fine frame  train_f1={tr_metrics['fine_frame_f1']:.4f}  val_f1={va_metrics['fine_frame_f1']:.4f}\n"
            f"  corr onset  train_f1={tr_metrics['corr_on_f1']:.4f}  val_f1={va_metrics['corr_on_f1']:.4f}\n"
            f"  note         train_f1={tr_note['note_f1']:.4f}  val_f1={va_note['note_f1']:.4f}"
        )

        # ── Wandb log ─────────────────────────────────────────────────────────
        log_dict: Dict[str, float] = {"epoch": epoch, "lr": current_lr}
        log_dict["train/loss"] = tr_loss
        log_dict["val/loss"]   = va_loss

        for k, v in tr_metrics.items():
            log_dict[f"train/{k}"] = v
        for k, v in va_metrics.items():
            log_dict[f"val/{k}"] = v
        for k, v in tr_note.items():
            log_dict[f"train/{k}"] = v
        for k, v in va_note.items():
            log_dict[f"val/{k}"] = v

        wandb.log(log_dict, step=epoch)

        # ── Checkpoint ───────────────────────────────────────────────────────
        all_metrics = {**{f"train_{k}": v for k, v in tr_metrics.items()},
                       **{f"val_{k}":   v for k, v in va_metrics.items()},
                       "train_loss": tr_loss, "val_loss": va_loss}
        save_checkpoint(model, optimizer, scheduler, epoch,
                        all_metrics, args.checkpoint_dir, config_snapshot)

        # ── Visualization ─────────────────────────────────────────────────────
        if len(val_ds) > 0:
            log_visualizations(model, val_ds, epoch, device, args.threshold, np_rng)

        # Bound memory growth across epochs without per-batch overhead.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
