"""
refine_experiment.py — Full training + validation pipeline for FineAMT.

Usage:
    python experiment/refine_experiment.py \
        --list_dir      dataset/corpus/MAESTRO-V3/list \
        --feature_dir   dataset/corpus/MAESTRO-V3/feature \
        --midi_dir      dataset/corpus/MAESTRO-V3/midi \
        --midi_cache_dir dataset/corpus/MAESTRO-V3/norm \
        --checkpoint_dir checkpoints/refine \
        --wandb_project  refine-amt \
        --epochs 20
"""

from __future__ import annotations

import argparse
import os
import pickle
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
# Metadata builder
# ============================================================================

def build_metadata(
    list_dir: str,
    feature_dir: str,
    midi_dir: str,
    midi_cache_dir: str,
    num_frame: int,
) -> Dict[str, List[Dict]]:
    """
    Build RefineDataset metadata dicts, grouped by split.

    Mirrors the window-indexing logic of cache_spec.build_midi_index so that
    global window indices match the midi_N.pkl / ts_target_N.pkl files written
    by cache_spec.py.

    Returns:
        {"train": [...], "test": [...], "valid": [...]}
        each entry is a dict with keys:
            midi_path, ts_target_path, orig_spec_path, start_frame, end_frame
    """
    split_meta: Dict[str, List[Dict]] = {"train": [], "test": [], "valid": []}
    window_idx = 0

    for split in ("train", "test", "valid"):
        list_path = os.path.join(list_dir, f"{split}.list")
        if not os.path.exists(list_path):
            continue

        with open(list_path, "r", encoding="utf-8") as fh:
            fnames = [line.strip() for line in fh if line.strip()]

        for fname in fnames:
            spec_path = os.path.join(feature_dir, fname + ".pkl")
            midi_path_orig = os.path.join(midi_dir, fname + ".mid")

            # Must mirror cache_spec.build_midi_index: skip if either is missing
            if not (os.path.exists(spec_path) and os.path.exists(midi_path_orig)):
                continue

            # Load spec to count frames without keeping data in memory
            try:
                with open(spec_path, "rb") as fh:
                    spec_raw = pickle.load(fh)
                if isinstance(spec_raw, (tuple, list)):
                    spec_raw = spec_raw[0]
                if isinstance(spec_raw, torch.Tensor):
                    T = spec_raw.shape[-1]
                else:
                    T = int(np.array(spec_raw).shape[-1])
                del spec_raw
            except Exception as exc:
                print(f"  Skipping {fname}: {exc}")
                continue

            for s in range(0, T, num_frame):
                e = min(s + num_frame, T)
                cache_midi = os.path.join(midi_cache_dir, f"midi_{window_idx}.pkl")
                cache_ts   = os.path.join(midi_cache_dir, f"ts_target_{window_idx}.pkl")

                if os.path.exists(cache_midi) and os.path.exists(cache_ts):
                    split_meta[split].append({
                        "midi_path":      cache_midi,
                        "ts_target_path": cache_ts,
                        "orig_spec_path": spec_path,
                        "start_frame":    s,
                        "end_frame":      e,
                    })

                window_idx += 1

    for split, meta in split_meta.items():
        print(f"  {split}: {len(meta)} windows")

    return split_meta


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


def _pair_notes(on_bin: np.ndarray, off_bin: np.ndarray) -> List[Tuple[int, int]]:
    """
    Greedily pair onset positions with the nearest following offset.

    Args:
        on_bin, off_bin : 1-D bool arrays of length T (one pitch)

    Returns:
        List of (onset_frame, offset_frame) pairs
    """
    onsets  = np.where(on_bin)[0].tolist()
    offsets = np.where(off_bin)[0].tolist()
    notes: List[Tuple[int, int]] = []
    off_ptr = 0
    for on in onsets:
        # advance offset pointer past the onset
        while off_ptr < len(offsets) and offsets[off_ptr] < on:
            off_ptr += 1
        off = offsets[off_ptr] if off_ptr < len(offsets) else on
        notes.append((on, off))
    return notes


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
    on_p  = (on_pred  > threshold).numpy()
    off_p = (off_pred > threshold).numpy()
    on_g  = (on_gt    > 0.5).numpy()
    off_g = (off_gt   > 0.5).numpy()

    total_tp = total_fp = total_fn = 0

    for pitch in range(on_p.shape[1]):
        pred_notes = _pair_notes(on_p[:, pitch],  off_p[:, pitch])
        gt_notes   = _pair_notes(on_g[:, pitch],  off_g[:, pitch])

        matched_gt = set()
        tp = 0
        for (p_on, p_off) in pred_notes:
            for gi, (g_on, g_off) in enumerate(gt_notes):
                if gi in matched_gt:
                    continue
                if abs(p_on - g_on) <= tolerance and abs(p_off - g_off) <= tolerance:
                    tp += 1
                    matched_gt.add(gi)
                    break

        fp = len(pred_notes) - tp
        fn = len(gt_notes)   - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / (total_tp + total_fp + 1e-7)
    rec  = total_tp / (total_tp + total_fn + 1e-7)
    f1   = 2.0 * prec * rec / (prec + rec + 1e-7)
    return {"note_precision": prec, "note_recall": rec, "note_f1": f1}


# ============================================================================
# Training / Validation
# ============================================================================

def _extract_beat_tensors(
    fine_out:    Dict[str, torch.Tensor],
    batch:       Dict[str, object],
    device:      torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Extract beat-position predictions and GT targets as flat [N, 128] tensors.

    beat_mask [B, max_len] selects every type_id==1 position in the batch.
    midi_mask [B, max_x]   selects valid (non-padded) GT entries.
    Both have the same total True count, so we can zip them.
    """
    beat_mask: torch.Tensor = batch["type_ids"].to(device) == 1   # [B, max_len]
    midi_mask: torch.Tensor = batch["midi_mask"].to(device)        # [B, max_x]

    preds: Dict[str, torch.Tensor]   = {}
    targets: Dict[str, torch.Tensor] = {}

    for k in ("on", "off", "frame"):
        preds[k]   = fine_out[k][beat_mask]                        # [N, 128]
        targets[k] = batch["midi_labels"][k].to(device)[midi_mask] # [N, 128]

    return preds, targets


def train_one_epoch(
    model:      FineAMT,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    scheduler:  Optional[object] = None,
    scaler:     Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    One training epoch.

    Returns:
        avg_loss   : scalar float
        all_preds  : {k: [N_epoch, 128]}  on CPU
        all_targets: {k: [N_epoch, 128]}  on CPU
    """
    model.train()
    total_loss  = 0.0
    n_samples   = 0

    all_preds:   Dict[str, List[torch.Tensor]] = {"on": [], "off": [], "frame": []}
    all_targets: Dict[str, List[torch.Tensor]] = {"on": [], "off": [], "frame": []}

    use_amp = scaler is not None

    for batch in tqdm(loader, desc="train", leave=False):
        seq      = batch["sequence"].to(device)
        type_ids = batch["type_ids"].to(device)

        model_batch = {"sequence": seq, "type_ids": type_ids}

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                fine_out = model(model_batch)
                preds, targets = _extract_beat_tensors(fine_out, batch, device)
                loss = sum(
                    F.binary_cross_entropy(preds[k], targets[k])
                    for k in ("on", "off", "frame")
                )
        else:
            fine_out = model(model_batch)
            preds, targets = _extract_beat_tensors(fine_out, batch, device)
            loss = sum(
                F.binary_cross_entropy(preds[k], targets[k])
                for k in ("on", "off", "frame")
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

        n = preds["on"].shape[0]
        total_loss += loss.item() * n
        n_samples  += n

        for k in ("on", "off", "frame"):
            all_preds[k].append(preds[k].detach().cpu())
            all_targets[k].append(targets[k].detach().cpu())

    avg_loss   = total_loss / max(1, n_samples)
    epoch_preds   = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
    epoch_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
    return avg_loss, epoch_preds, epoch_targets


@torch.no_grad()
def validate_one_epoch(
    model:  FineAMT,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    One validation epoch (no gradients).

    Returns:
        avg_loss   : scalar float
        all_preds  : {k: [N_epoch, 128]}  on CPU
        all_targets: {k: [N_epoch, 128]}  on CPU
    """
    model.eval()
    total_loss  = 0.0
    n_samples   = 0

    all_preds:   Dict[str, List[torch.Tensor]] = {"on": [], "off": [], "frame": []}
    all_targets: Dict[str, List[torch.Tensor]] = {"on": [], "off": [], "frame": []}

    for batch in tqdm(loader, desc="val  ", leave=False):
        seq      = batch["sequence"].to(device)
        type_ids = batch["type_ids"].to(device)

        fine_out = model({"sequence": seq, "type_ids": type_ids})
        preds, targets = _extract_beat_tensors(fine_out, batch, device)

        loss = sum(
            F.binary_cross_entropy(preds[k], targets[k])
            for k in ("on", "off", "frame")
        )

        n = preds["on"].shape[0]
        total_loss += loss.item() * n
        n_samples  += n

        for k in ("on", "off", "frame"):
            all_preds[k].append(preds[k].cpu())
            all_targets[k].append(targets[k].cpu())

    avg_loss      = total_loss / max(1, n_samples)
    epoch_preds   = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
    epoch_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
    return avg_loss, epoch_preds, epoch_targets


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
    batch = collate_refine([sample])
    seq      = batch["sequence"].to(device)
    type_ids = batch["type_ids"].to(device)

    with torch.no_grad():
        fine_out = model({"sequence": seq, "type_ids": type_ids})

    # ── spectrogram (spec tokens in chronological order) ──────────────────
    spec_mask = (type_ids[0] == 0)                     # [L]
    spec_tok  = seq[0][spec_mask].cpu().numpy()        # [T, D]
    # Use only the first n_mels dims for display
    n_mels    = min(128, spec_tok.shape[1])
    spec_img  = spec_tok[:, :n_mels].T                 # [n_mels, T]

    # ── GT labels (beat tokens) ────────────────────────────────────────────
    midi_mask = batch["midi_mask"][0]                  # [max_x]
    gt_on  = batch["midi_labels"]["on"][0][midi_mask].numpy().T    # [128, x]
    gt_off = batch["midi_labels"]["off"][0][midi_mask].numpy().T
    gt_frm = batch["midi_labels"]["frame"][0][midi_mask].numpy().T

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
    parser.add_argument("--list_dir",       required=True,  help="Dir with train/test/valid .list files")
    parser.add_argument("--feature_dir",    required=True,  help="Spectrogram pickle dir  ({fname}.pkl)")
    parser.add_argument("--midi_dir",       required=True,  help="Original MIDI dir  ({fname}.mid)")
    parser.add_argument("--midi_cache_dir", required=True,  help="cache_spec.py output dir (midi_N.pkl)")
    parser.add_argument("--checkpoint_dir", default="checkpoints/refine", help="Checkpoint output dir")
    # Dataset
    parser.add_argument("--num_frame", type=int, default=128,  help="Frames per spectrogram window")
    parser.add_argument("--n_mels",    type=int, default=128,  help="Mel bins in spectrogram")
    parser.add_argument("--dt",        type=float, default=256/16000, help="Seconds per spec frame")
    parser.add_argument("--cache_size",type=int, default=512,  help="Pickle LRU cache size")
    # Model
    parser.add_argument("--blocks", type=int, default=6,   help="Mamba2 block count")
    parser.add_argument("--dim",    type=int, default=256, help="Model d_model")
    parser.add_argument("--d_state",type=int, default=64)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--max_len",type=int, default=4096)
    # Training
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--optimizer",  default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler",  default="cosine",
                        choices=["onecycle", "cosine", "constant", "linear", "exponential"])
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--num_workers",type=int,   default=0)
    parser.add_argument("--amp",        action="store_true", help="Use automatic mixed precision")
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
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Building metadata...")
    split_meta = build_metadata(
        list_dir      = args.list_dir,
        feature_dir   = args.feature_dir,
        midi_dir      = args.midi_dir,
        midi_cache_dir= args.midi_cache_dir,
        num_frame     = args.num_frame,
    )

    train_ds = RefineDataset(
        metadata   = split_meta["train"],
        n_mels     = args.n_mels,
        dt         = args.dt,
        cache_size = args.cache_size,
    )
    val_ds = RefineDataset(
        metadata   = split_meta["valid"],
        n_mels     = args.n_mels,
        dt         = args.dt,
        cache_size = args.cache_size,
    )
    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    loader_kwargs = dict(
        batch_size  = args.batch_size,
        collate_fn  = collate_refine,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        persistent_workers = args.num_workers > 0,
    )
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
        tr_loss, tr_preds, tr_targets = train_one_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler, scaler=scaler,
        )
        tr_metrics = compute_metrics(tr_preds, tr_targets, args.threshold)
        tr_note    = compute_note_f1(
            tr_preds["on"], tr_preds["off"],
            tr_targets["on"], tr_targets["off"],
            threshold = args.threshold,
        )

        # ── Validate ─────────────────────────────────────────────────────────
        va_loss, va_preds, va_targets = validate_one_epoch(model, val_loader, device)
        va_metrics = compute_metrics(va_preds, va_targets, args.threshold)
        va_note    = compute_note_f1(
            va_preds["on"], va_preds["off"],
            va_targets["on"], va_targets["off"],
            threshold = args.threshold,
        )

        # ── LR ───────────────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Print summary ────────────────────────────────────────────────────
        print(
            f"  loss  train={tr_loss:.4f}  val={va_loss:.4f}  lr={current_lr:.2e}\n"
            f"  onset  train_f1={tr_metrics['on_f1']:.4f}  val_f1={va_metrics['on_f1']:.4f}\n"
            f"  frame  train_f1={tr_metrics['frame_f1']:.4f}  val_f1={va_metrics['frame_f1']:.4f}\n"
            f"  note   train_f1={tr_note['note_f1']:.4f}  val_f1={va_note['note_f1']:.4f}"
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

    wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
