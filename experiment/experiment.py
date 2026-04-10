"""Training and evaluation pipeline for the FineAMT beat-synchronous AMT model."""

import os
import sys
import math
import pickle
import random
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pretty_midi import PrettyMIDI

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util as _ilu
import types as _types

def _load_config():
    _spec = _ilu.spec_from_file_location(
        "experiment_config",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod

cfg = _load_config()
from models.fine import FineAMT, collate_refine
from dataset.refine_dataset import RefineDataset
from components.schedulers import make_optimizer, make_scheduler
from utils.beat_utils import get_beats_and_downbeats

# Label keys used for loss and frame-level metrics (excludes velocity)
LABEL_KEYS = ("on", "off", "frame")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_device(batch: Dict, device: str) -> Dict:
    """Recursively move tensors in a collated batch dict to *device*."""
    out: Dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


def compute_fine_loss(fine_out: Dict[str, torch.Tensor], batch: Dict) -> torch.Tensor:
    """BCE loss on fine-head predictions (on, off, frame) against normalized_labels.

    Sigmoid is already applied by OutputHead, so we use F.binary_cross_entropy
    with clamped predictions for numerical stability.
    """
    type_ids = batch["type_ids"]               # [B, max_len]
    norm_labels = batch["normalized_labels"]    # {k: [B, L, 128]}
    fine_mask = (type_ids == 1)                 # [B, max_len]
    B = type_ids.shape[0]

    total_loss = type_ids.new_tensor(0.0, dtype=torch.float32)
    count = 0

    for k in LABEL_KEYS:
        preds_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        for i in range(B):
            mask_i = fine_mask[i]
            x_i = mask_i.sum()
            if x_i == 0:
                continue
            preds_list.append(fine_out[k][i, mask_i])        # [x_i, 128]
            targets_list.append(norm_labels[k][i, :x_i])     # [x_i, 128]

        if preds_list:
            p = torch.cat(preds_list).clamp(1e-7, 1.0 - 1e-7)
            t = torch.cat(targets_list)
            total_loss = total_loss + F.binary_cross_entropy(p, t, reduction="sum")
            count += p.numel()

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Note extraction & matching
# ---------------------------------------------------------------------------

def _extract_notes(
    onsets: np.ndarray,
    frames: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, int]]:
    """Return ``[(pitch, start_frame, end_frame), ...]`` from onset/frame arrays.

    Args:
        onsets: ``[T, 128]`` onset probabilities.
        frames: ``[T, 128]`` frame probabilities.
        threshold: detection threshold applied to both arrays.
    """
    notes: List[Tuple[int, int, int]] = []
    T, P = onsets.shape
    for p in range(P):
        on_indices = np.where(onsets[:, p] > threshold)[0]
        if len(on_indices) == 0:
            continue
        for i, on_t in enumerate(on_indices):
            next_onset = int(on_indices[i + 1]) if i + 1 < len(on_indices) else T
            inactive = np.where(frames[on_t:next_onset, p] <= threshold)[0]
            off_t = int(on_t + inactive[0]) if len(inactive) > 0 else next_onset
            if off_t > on_t:
                notes.append((p, int(on_t), off_t))
    return notes


def _match_notes(
    pred_notes: List[Tuple[int, int, int]],
    target_notes: List[Tuple[int, int, int]],
    onset_tolerance: int = 3,
) -> int:
    """Greedy onset+pitch matching.  Returns the number of true-positive matches."""
    matched_pred: set = set()
    tp = 0
    for t_pitch, t_start, _t_end in sorted(target_notes, key=lambda n: n[1]):
        best_idx = None
        best_dist = onset_tolerance + 1
        for pi, (p_pitch, p_start, _p_end) in enumerate(pred_notes):
            if pi in matched_pred or p_pitch != t_pitch:
                continue
            d = abs(p_start - t_start)
            if d <= onset_tolerance and d < best_dist:
                best_dist = d
                best_idx = pi
        if best_idx is not None:
            matched_pred.add(best_idx)
            tp += 1
    return tp


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1 from raw counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """Accumulates frame-level and note-level TP/FP/FN across batches."""

    def __init__(self, threshold: float = 0.5, onset_tolerance: int = 3) -> None:
        self.threshold = threshold
        self.onset_tolerance = onset_tolerance
        self.frame_counts: Dict[str, Dict[str, int]] = {
            k: {"tp": 0, "fp": 0, "fn": 0} for k in LABEL_KEYS
        }
        self.note_tp = 0
        self.note_fp = 0
        self.note_fn = 0

    def update(self, fine_out: Dict[str, torch.Tensor], batch: Dict) -> None:
        type_ids = batch["type_ids"]               # [B, max_len]
        norm_labels = batch["normalized_labels"]    # {k: [B, L, 128]}
        fine_mask = (type_ids == 1)
        B = type_ids.shape[0]

        for i in range(B):
            mask_i = fine_mask[i]
            x_i = mask_i.sum().item()
            if x_i == 0:
                continue

            for k in LABEL_KEYS:
                pred = fine_out[k][i, mask_i].detach().cpu().numpy()   # [x_i, 128]
                target = norm_labels[k][i, :x_i].detach().cpu().numpy()

                pred_bin = (pred > self.threshold).astype(np.float32)
                tgt_bin = (target > self.threshold).astype(np.float32)

                self.frame_counts[k]["tp"] += int((pred_bin * tgt_bin).sum())
                self.frame_counts[k]["fp"] += int((pred_bin * (1.0 - tgt_bin)).sum())
                self.frame_counts[k]["fn"] += int(((1.0 - pred_bin) * tgt_bin).sum())

            # Note-level
            pred_on = fine_out["on"][i, mask_i].detach().cpu().numpy()
            pred_fr = fine_out["frame"][i, mask_i].detach().cpu().numpy()
            tgt_on = norm_labels["on"][i, :x_i].detach().cpu().numpy()
            tgt_fr = norm_labels["frame"][i, :x_i].detach().cpu().numpy()

            pred_notes = _extract_notes(pred_on, pred_fr, self.threshold)
            tgt_notes = _extract_notes(tgt_on, tgt_fr, self.threshold)

            tp = _match_notes(pred_notes, tgt_notes, self.onset_tolerance)
            self.note_tp += tp
            self.note_fp += len(pred_notes) - tp
            self.note_fn += len(tgt_notes) - tp

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for k in LABEL_KEYS:
            c = self.frame_counts[k]
            p, r, f = _prf(c["tp"], c["fp"], c["fn"])
            metrics[f"{k}_precision"] = p
            metrics[f"{k}_recall"] = r
            metrics[f"{k}_f1"] = f

        p, r, f = _prf(self.note_tp, self.note_fp, self.note_fn)
        metrics["note_precision"] = p
        metrics["note_recall"] = r
        metrics["note_f1"] = f
        return metrics


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: FineAMT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    accumulator: MetricAccumulator,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        batch = to_device(batch, device)
        _, fine_out = model(batch)
        loss = compute_fine_loss(fine_out, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            accumulator.update(fine_out, batch)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_one_epoch(
    model: FineAMT,
    loader: DataLoader,
    device: str,
    accumulator: MetricAccumulator,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        batch = to_device(batch, device)
        _, fine_out = model(batch)
        loss = compute_fine_loss(fine_out, batch)
        total_loss += loss.item()
        n_batches += 1
        accumulator.update(fine_out, batch)

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: FineAMT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def log_visualizations(
    model: FineAMT,
    val_dataset: RefineDataset,
    epoch: int,
    device: str,
    threshold: float,
) -> None:
    """Pick a random validation sample, run inference, and log GT-vs-pred
    images to wandb."""
    model.eval()
    idx = random.randint(0, len(val_dataset) - 1)
    sample = val_dataset[idx]
    meta = val_dataset.metadata[idx]

    # Load raw spectrogram (handle tuple/list pickle format from conv_wav2fe)
    with open(meta["orig_spec_path"], "rb") as fh:
        spec_raw = pickle.load(fh)
    if isinstance(spec_raw, (tuple, list)):
        spec_raw = spec_raw[0]
    if isinstance(spec_raw, torch.Tensor):
        spec_raw = spec_raw.numpy()
    elif not isinstance(spec_raw, np.ndarray):
        spec_raw = np.array(spec_raw, dtype=np.float32)
    if spec_raw.ndim == 3:
        spec_raw = spec_raw.squeeze(0)              # [n_mels, T_full]

    s = int(meta["start_frame"])
    e = int(meta["end_frame"])
    spec_window = spec_raw[:, s:e]                   # [n_mels, T]

    # Forward pass on single sample
    batch = collate_refine([sample])
    batch = to_device(batch, device)
    with torch.no_grad():
        _, fine_out = model(batch)

    type_ids = batch["type_ids"][0]                  # [max_len]
    mask = (type_ids == 1)
    x = mask.sum().item()

    # Build 7-panel figure: spec + 3x(GT, Pred) for onset / frame / offset
    fig, axes = plt.subplots(7, 1, figsize=(16, 20))

    axes[0].imshow(spec_window, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Spectrogram (raw window)")
    axes[0].set_ylabel("Mel bin")

    panel_info = [("on", "Onset"), ("frame", "Frame"), ("off", "Offset")]
    for i, (k, name) in enumerate(panel_info):
        gt = batch["normalized_labels"][k][0, :x].cpu().numpy()
        pred = fine_out[k][0, mask].cpu().numpy()

        ax_gt = axes[1 + 2 * i]
        ax_gt.imshow(gt.T, aspect="auto", origin="lower", cmap="Oranges", vmin=0, vmax=1)
        ax_gt.set_title(f"GT {name}")
        ax_gt.set_ylabel("Pitch")

        ax_pr = axes[2 + 2 * i]
        pred_bin = (pred > threshold).astype(np.float32)
        ax_pr.imshow(pred_bin.T, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax_pr.set_title(f"Pred {name} (thr={threshold})")
        ax_pr.set_ylabel("Pitch")

    axes[-1].set_xlabel("Beat-grid step")
    plt.tight_layout()
    wandb.log({"val/visualization": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metadata builder helpers
# ---------------------------------------------------------------------------

def _parse_midi_file(
    args: Tuple[str, str, str, str, int, int],
) -> Tuple[str, str, str, Optional[List[int]]]:
    """Parse one MIDI file and return its window start indices.

    Designed to run in a worker process.  Returns
    ``(fname, spec_path, label_path, starts_or_None)``.
    """
    fname, midi_path, spec_path, label_path, beats_per_window, hop_beats = args
    try:
        pm = PrettyMIDI(midi_path)
    except Exception:
        return fname, spec_path, label_path, None

    beats, _ = get_beats_and_downbeats(pm)
    K = max(0, len(beats) - 1)
    if K < beats_per_window:
        return fname, spec_path, label_path, None

    starts = list(range(0, K - beats_per_window + 1, hop_beats))
    last_start = K - beats_per_window
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return fname, spec_path, label_path, starts


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_metadata(
    list_dir: str,
    d_feature: str,
    d_label: str,
    d_midi: str,
    cache_dir: str,
    *,
    beats_per_window: int,
    hop_beats: int,
    dt: float,
    workers: int = 4,
) -> Dict[str, List[Dict]]:
    """Build per-window metadata dicts for :class:`RefineDataset`.

    Iterates train -> test -> valid ``.list`` files in the **same order** as
    ``cache_spec.py`` so that the global window index matches the cached
    ``labels_N / extra_N`` pickle numbering.

    Phase 1 (parallel): MIDI parsing + beat computation for each file.
    Phase 2 (sequential): window-index assignment + extra pickle loading.
    """
    result: Dict[str, List[Dict]] = {"train": [], "test": [], "valid": []}
    window_idx = 0

    for split in ("train", "test", "valid"):
        list_path = os.path.join(list_dir, f"{split}.list")
        if not os.path.exists(list_path):
            continue

        with open(list_path, "r", encoding="utf-8") as fh:
            fnames = [line.rstrip("\n") for line in fh if line.strip()]

        # --- Phase 1: parallel MIDI parsing ---
        parse_args = [
            (
                fname,
                os.path.join(d_midi, fname + ".mid"),
                os.path.join(d_feature, fname + ".pkl"),
                os.path.join(d_label, fname + ".pkl"),
                beats_per_window,
                hop_beats,
            )
            for fname in fnames
            if (
                os.path.exists(os.path.join(d_midi, fname + ".mid"))
                and os.path.exists(os.path.join(d_feature, fname + ".pkl"))
                and os.path.exists(os.path.join(d_label, fname + ".pkl"))
            )
        ]

        parsed: Dict[str, Tuple[str, str, Optional[List[int]]]] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fname, spec_path, label_path, starts in tqdm(
                pool.map(_parse_midi_file, parse_args),
                total=len(parse_args),
                desc=f"  {split} (parse)",
                unit="file",
            ):
                parsed[fname] = (spec_path, label_path, starts)

        # --- Phase 2: sequential index assignment + pickle loading ---
        for fname in tqdm(fnames, desc=f"  {split} (index)", unit="file"):
            if fname not in parsed:
                # file was missing on disk — skip without advancing window_idx
                # (cache_spec.py also skips missing files)
                continue

            spec_path, label_path, starts = parsed[fname]
            if starts is None:
                continue

            for _start_beat_idx in starts:
                extra_path = os.path.join(cache_dir, f"extra_{window_idx}.pkl")
                norm_labels_path = os.path.join(cache_dir, f"labels_{window_idx}.pkl")

                if not os.path.exists(extra_path):
                    window_idx += 1
                    continue

                try:
                    with open(extra_path, "rb") as fh:
                        extra = pickle.load(fh)
                except (EOFError, pickle.UnpicklingError):
                    window_idx += 1
                    continue

                times = extra["times"]
                max_time = extra["max_time"]

                t_first = (
                    times[0].item()
                    if isinstance(times, torch.Tensor)
                    else float(times[0])
                )
                t_last = (
                    times[-1].item()
                    if isinstance(times, torch.Tensor)
                    else float(times[-1])
                )

                start_frame = max(0, int(math.floor(t_first / dt)))
                if max_time is not None:
                    end_frame = int(math.ceil(float(max_time) / dt))
                else:
                    end_frame = int(math.ceil(t_last / dt)) + 1

                result[split].append(
                    {
                        "norm_labels_path": norm_labels_path,
                        "extra_path": extra_path,
                        "orig_spec_path": spec_path,
                        "orig_labels_path": label_path,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
                window_idx += 1

    return result


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(
    model: FineAMT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: RefineDataset,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler,
    num_epochs: int,
    device: str,
    checkpoint_dir: str,
    threshold: float,
    onset_tolerance: int = 3,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_note_f1 = -1.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        # --- Train ---
        train_acc = MetricAccumulator(threshold, onset_tolerance)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, train_acc,
        )
        train_metrics = train_acc.compute()

        # --- Validate ---
        val_acc = MetricAccumulator(threshold, onset_tolerance)
        val_loss = validate_one_epoch(model, val_loader, device, val_acc)
        val_metrics = val_acc.compute()

        lr = optimizer.param_groups[0]["lr"]

        # --- Console summary ---
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.2e}")
        for tag, m in [("train", train_metrics), ("val", val_metrics)]:
            print(
                f"  {tag}: on_f1={m['on_f1']:.3f}  frame_f1={m['frame_f1']:.3f}  "
                f"off_f1={m['off_f1']:.3f}  note_f1={m['note_f1']:.3f}"
            )

        # --- wandb ---
        log_dict: Dict[str, float] = {
            "epoch": float(epoch),
            "lr": lr,
            "train/loss": train_loss,
            "val/loss": val_loss,
        }
        for tag, m in [("train", train_metrics), ("val", val_metrics)]:
            for mk, mv in m.items():
                log_dict[f"{tag}/{mk}"] = mv
        wandb.log(log_dict, step=epoch)

        # --- Visualization ---
        log_visualizations(model, val_dataset, epoch, device, threshold)

        # --- Checkpoint every epoch ---
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path)

        # --- Milestone every 5 epochs ---
        if epoch % 5 == 0:
            mile_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch}_milestone.pt",
            )
            save_checkpoint(model, optimizer, epoch, val_metrics, mile_path)

        # --- Best model ---
        if val_metrics["note_f1"] > best_note_f1:
            best_note_f1 = val_metrics["note_f1"]
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path)
            print(f"  ** New best model (note_f1={best_note_f1:.4f}) **")

    print(f"\nTraining complete.  Best val note_f1: {best_note_f1:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FineAMT")
    parser.add_argument("-d_list", required=True,
                        help="Dir with train/test/valid .list files")
    parser.add_argument("-d_feature", required=True,
                        help="Pre-computed spectrogram pickles dir")
    parser.add_argument("-d_label", required=True,
                        help="Pre-computed label pickles dir")
    parser.add_argument("-d_midi", required=True,
                        help="MIDI files dir")
    parser.add_argument("-d_cache", required=True,
                        help="Beat-normalised cache dir (extra/labels pickles)")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--scheduler", type=str, default="onecycle")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--wandb_project", type=str, default="fine-amt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--metadata_workers", type=int, default=4,
                        help="Worker processes for parallel MIDI parsing")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = cfg.DEVICE

    # Derived constants from config
    dt = cfg.coarse_spectrogram["HOP_LENGTH"] / cfg.coarse_spectrogram["SAMPLE_RATE"]
    beats_per_window = cfg.beat_normalized_spectrogram["BEATS_PER_CLIP"]
    hop_beats = beats_per_window  # non-overlapping windows

    # Build metadata
    print("Building metadata ...")
    metadata = build_metadata(
        args.d_list,
        args.d_feature,
        args.d_label,
        args.d_midi,
        args.d_cache,
        beats_per_window=beats_per_window,
        hop_beats=hop_beats,
        dt=dt,
        workers=args.metadata_workers,
    )
    for sp in ("train", "test", "valid"):
        print(f"  {sp}: {len(metadata[sp])} windows")

    # Datasets & loaders
    n_mels = cfg.coarse_spectrogram["N_MELS"]
    train_ds = RefineDataset(
        metadata["train"], feature_dim=args.feature_dim, n_mels=n_mels, dt=dt,
    )
    val_ds = RefineDataset(
        metadata["valid"], feature_dim=args.feature_dim, n_mels=n_mels, dt=dt,
    )

    persistent = cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_refine,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=cfg.DROP_LAST_TRAIN,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_refine,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
        persistent_workers=persistent,
    )

    # Model
    model = FineAMT(
        blocks=args.blocks,
        dim=args.dim,
        feature_dim=args.feature_dim,
    ).to(device)

    # Optimizer & scheduler
    optimizer = make_optimizer(
        model, optimizer_type="adam", lr=args.lr, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = make_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1),
    )

    # wandb
    wandb.init(
        project=args.wandb_project,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "blocks": args.blocks,
            "dim": args.dim,
            "feature_dim": args.feature_dim,
            "scheduler": args.scheduler,
            "threshold": args.threshold,
            "seed": args.seed,
            "beats_per_window": beats_per_window,
            "weight_decay": cfg.WEIGHT_DECAY,
        },
    )

    # Train
    run(
        model,
        train_loader,
        val_loader,
        val_ds,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        threshold=args.threshold,
    )

    wandb.finish()