# test_v2.8.py – PyTorch 2.8.0 / multi‑GPU rewrite of test.py
# -----------------------------------------------------------
#  * Distributed training with DDP (two RTX‑5090 GPUs target).
#  * Maintains original logic & behaviour exactly.
#  * Updated deprecated/removed APIs, modern dataloader/scheduler use.
#  * Designed to be launched via `torchrun --nproc_per_node=2 test_v2.8.py`.
import argparse
import os
import datetime
import json
import random
import platform
import time
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pretty_midi import PrettyMIDI

from dataset.cache_norm import cache_data
from dataset.transforms import MelTransform
from dataset.cashe_dataset import CashDataset
from utils.display_midi import display_spectrogram, display_midi_from_roll
from components.loss import OnsetsAndFramesLoss, OnsetsAndFramesPaperLoss
from models.hFT import HFTModel
from components.schedulers import make_optimizer, make_scheduler
from experiment.config import (CSV_PATH, N_MELS, CACHE_PATH, BATCH_SIZE, DEVICE,
                    CSV_PATH, MAESTRO_ROOT, BEATS_PER_CLIP, NUM_EPOCHS,
                    SUBDIVISIONS_PER_BEAT, HOP_LENGTH, SAMPLE_RATE,)

LABEL_KEYS = ("on", "off", "frame", "vel")

# AMP / compile detection (unchanged but adapted for 2.8)
# AMP is off by default to match original `test.py`.  Use the
# --amp command‑line flag to enable it (see main()).
USE_AMP = False
USE_COMPILE = hasattr(torch, "compile")
try:
    import triton  # compile requires Triton 2.0+
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
USE_COMPILE = USE_COMPILE and TRITON_AVAILABLE


def mark_cudagraph_step_begin():
    """Mark a new CUDA Graph step when supported by this PyTorch build."""
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def clone_model_outputs(out_dict):
    """Detach+clone model outputs for safe use outside the current compiled step."""
    return {k: v.detach().clone() for k, v in out_dict.items()}


def clone_model_outputs_for_backward(out_dict):
    """Clone model outputs while preserving autograd links for backward."""
    return {k: v.clone() for k, v in out_dict.items()}

# --- logging/checkpoint utilities ------------------------------------------------
def setup_run_dir(run_name=None):
    """Create directory structure for a new training run and return its path."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"run_{ts}"
    base = os.path.join("training_runs", run_name)
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base, "val_preds"), exist_ok=True)
    # config.json and metrics.jsonl will be written later
    return base


def log_metrics(metrics_path, entry):
    """Append a JSON line to the metrics log and flush to disk."""
    def _to_json_safe(v):
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return v.item()
            return v.detach().cpu().tolist()
        if isinstance(v, np.generic):
            return v.item()
        return v

    safe_entry = {k: _to_json_safe(v) for k, v in entry.items()}
    with open(metrics_path, "a") as f:
        f.write(json.dumps(safe_entry) + "\n")
        f.flush()


def save_checkpoint(run_dir, epoch, model, optimizer, scheduler, scaler,
                    global_step, is_best=False):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    fname = os.path.join(run_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt")
    torch.save(state, fname)
    if is_best:
        torch.save(state, os.path.join(run_dir, "best_model.pt"))


def compute_eval_metrics(model, loader, criterion, device, threshold=0.5):
    """Evaluate model on a dataset and return loss + F1/precision/recall stats."""
    model.eval()
    total_loss = 0.0
    nsteps = 0
    all_true_on, all_pred_on = [], []
    all_true_frame, all_pred_frame = [], []
    with torch.no_grad():
        for x, labels in loader:
            # Ensure input is on the correct device
            x = x.to(device, dtype=torch.float32)
            y = {}
            for k in LABEL_KEYS:
                if k in labels:
                    v = labels[k]
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    # Ensure label is on the correct device
                    y[k] = v.to(device=device, dtype=torch.float32)
            mark_cudagraph_step_begin()
            out = model(x)
            loss, _ = criterion(out, y)
            total_loss += loss.item()
            nsteps += 1

            pred_on = (torch.sigmoid(out["on"]) > threshold).cpu().flatten()
            true_on = y["on"].cpu().flatten()
            all_true_on.append(true_on.float())
            all_pred_on.append(pred_on.float())

            pred_frame = (torch.sigmoid(out["frame"]) > threshold).cpu().flatten()
            true_frame = y["frame"].cpu().flatten()
            all_true_frame.append(true_frame.float())
            all_pred_frame.append(pred_frame.float())

    avg_loss = total_loss / max(1, nsteps)
    def f1_stats(true, pred):
        if true.numel() == 0:
            return 0.0, 0.0, 0.0
        tp = (true * pred).sum().float()
        fp = ((1 - true) * pred).sum().float()
        fn = (true * (1 - pred)).sum().float()
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2 * prec * rec / (prec + rec + 1e-7)
        return f1.item(), prec.item(), rec.item()

    all_true_on = torch.cat(all_true_on) if all_true_on else torch.tensor([])
    all_pred_on = torch.cat(all_pred_on) if all_pred_on else torch.tensor([])
    all_true_frame = torch.cat(all_true_frame) if all_true_frame else torch.tensor([])
    all_pred_frame = torch.cat(all_pred_frame) if all_pred_frame else torch.tensor([])

    on_f1, on_prec, on_rec = f1_stats(all_true_on, all_pred_on)
    fr_f1, fr_prec, fr_rec = f1_stats(all_true_frame, all_pred_frame)

    return {
        "loss": avg_loss,
        "onset_f1": on_f1,
        "onset_prec": on_prec,
        "onset_rec": on_rec,
        "frame_f1": fr_f1,
        "frame_prec": fr_prec,
        "frame_rec": fr_rec,
    }

# ------------------------------------------------------------------------------

def load_model_checkpoint(model_path, device):
    """Load a saved model checkpoint (same as original)."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = HFTModel(dim=config["dim"],
                     depth=config.get("depth", 6),
                     num_heads=config["num_heads"],
                     n_pitches=config.get("n_pitches", 128))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"Loaded model from {model_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Final loss: {checkpoint['loss']:.6f}")
        print(f"  Config: {config}")
    return model

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None,
                    step_per_batch=True, device=None, use_amp=USE_AMP,
                    show_progress=True, accumulation_steps=1, epoch=None,
                    scaler=None):
    """Single‑epoch training; supports DDP sampler epoch setting.

    scaler: optional GradScaler instance that will be stepped/returned.  If
    None and `use_amp` is True, a new scaler will be created internally.
    """
    model.train()
    if device is None:
        # Default to model parameter device to avoid accidental CPU/CUDA mixing.
        device = next(model.parameters()).device
    total = 0.0
    lr_track = []
    num_samples = 0
    if use_amp:
        if scaler is None:
            # use new torch.amp API (warned in FutureWarning)
            scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    # advance distributed sampler epoch for shuffling
    if epoch is not None and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    total_batches = len(loader)
    accumulated_loss = 0.0
    accumulation_counter = 0
    grad_norms = []
    steps_done = 0

    for step, (x, labels) in enumerate(loader):
        if show_progress and step % 10 == 0:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"Batch {step+1}/{total_batches}", end="\r")

        batch_size = x.size(0)
        num_samples += batch_size

        x = x.to(device, dtype=torch.float32, non_blocking=True)

        # batch label transfer optimization (unchanged)
        y = {}
        label_tensors = []
        shapes_valid = True
        for k in LABEL_KEYS:
            if k in labels:
                v = labels[k]
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                label_tensors.append(v)
                if len(label_tensors) > 1 and v.shape != label_tensors[0].shape:
                    shapes_valid = False
        if label_tensors and shapes_valid:
            label_batch = torch.stack(label_tensors, dim=0).to(device=device,
                                                               dtype=torch.float32,
                                                               non_blocking=True)
            y = dict(zip(LABEL_KEYS, label_batch.unbind(0)))
        else:
            if not shapes_valid and label_tensors:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(
                        f"Warning: Label shapes don't match for batching: "
                        f"{[v.shape for v in label_tensors]}"
                    )
            for k in LABEL_KEYS:
                if k in labels:
                    v = labels[k]
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    y[k] = v.to(device=device, dtype=torch.float32, non_blocking=True)

        if use_amp:
            # use new autocast API
            with torch.amp.autocast('cuda'):
                mark_cudagraph_step_begin()
                out = model(x)
                out = clone_model_outputs_for_backward(out)
                loss, _metrics = criterion(out, y)
        else:
            mark_cudagraph_step_begin()
            out = model(x)
            out = clone_model_outputs_for_backward(out)
            loss, _metrics = criterion(out, y)

        loss = loss / accumulation_steps
        accumulated_loss += loss.item()

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulation_counter += 1

        if accumulation_counter % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None and step_per_batch:
                scheduler.step()

            grad_norms.append(grad_norm)
            steps_done += 1

            optimizer.zero_grad(set_to_none=True)
            total += accumulated_loss * batch_size * accumulation_steps
            accumulated_loss = 0.0

        lr_track.append(optimizer.param_groups[0]["lr"])

    # final accumulation step
    if accumulation_counter % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if scheduler is not None and step_per_batch:
            scheduler.step()
        grad_norms.append(grad_norm)
        steps_done += 1
        optimizer.zero_grad(set_to_none=True)
        total += accumulated_loss * batch_size * accumulation_steps

    if show_progress:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print()

    avg_loss = total / max(1, num_samples)
    avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    return avg_loss, lr_track, avg_grad, steps_done, scaler

# visualization helpers (unchanged)
def show_on_off_overlay(frm, S, pitch_lo=21, pitch_hi=108):
    frm_np = frm[pitch_lo:pitch_hi+1].cpu().numpy().T
    plt.figure(figsize=(12, 4))
    plt.imshow(frm_np, aspect="auto", origin="lower")
    for x in range(0, frm_np.shape[1], S):
        plt.axvline(x, linewidth=0.5, alpha=0.3)
    plt.xlabel("Subdivision index")
    plt.ylabel("Pitch index within range")
    plt.title("Frame probability distribution")
    plt.show()

def show_on_off_predictions(on_prob, off_prob, frame_prob,
                            threshold=0.5, S=24, pitch_lo=21, pitch_hi=108):
    on_pred = (torch.sigmoid(on_prob) > threshold).cpu().numpy()
    off_pred = (torch.sigmoid(off_prob) > threshold).cpu().numpy()
    frame_pred = (torch.sigmoid(frame_prob) > threshold).cpu().numpy()
    frame_pred = frame_pred[:, pitch_lo:pitch_hi+1].T
    on_pred = on_pred[:, pitch_lo:pitch_hi+1].T
    off_pred = off_pred[:, pitch_lo:pitch_hi+1].T
    plt.figure(figsize=(12, 4))
    plt.imshow(frame_pred, aspect="auto", origin="lower", cmap="gray", alpha=0.7)
    y_on, x_on = np.nonzero(on_pred)
    y_off, x_off = np.nonzero(off_pred)
    plt.scatter(x_on, y_on, s=8, marker="o", color="green", label="onset")
    plt.scatter(x_off, y_off, s=8, marker="x", color="red", label="offset")
    for x in range(0, frame_pred.shape[1], S):
        plt.axvline(x, linewidth=0.5, alpha=0.3)
    plt.legend(loc="upper right")
    plt.xlabel("Subdivision index")
    plt.ylabel("Pitch index within range")
    plt.title(
        "Predicted events: frames (gray), onsets (green), offsets (red)"
    )
    plt.show()

# The following functions (display_transformed_dataset, continue_training,
# test_model, compute_f1_score) remain unchanged from test.py except that they
# guard prints/saves with rank check when appropriate.  See original script.

def display_transformed_dataset(index):
    with open(f"{CACHE_PATH}//labels_{index}.pkl", "rb") as file:
        label = pickle.load(file)
        frm = label["frame"]
        show_on_off_overlay(frm, 12, 21, 108)
    with open(f"{CACHE_PATH}//spectrogram_{index}.pkl", "rb") as file:
        test_spec = pickle.load(file)
        display_spectrogram(test_spec)

def continue_training(model_path, additional_epochs, device):
    model = load_model_checkpoint(model_path, device)
    optimizer = make_optimizer(model, lr=1e-4)
    cache_dataset = CashDataset(CACHE_PATH)
    sampler = DistributedSampler(cache_dataset) if torch.distributed.is_initialized() else None
    cache_loader = DataLoader(
        cache_dataset,
        batch_size=16,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    criterion = OnsetsAndFramesLoss(
        lambda_on=0.5,
        lambda_frame=1.0,
        lambda_off=0.5,
        pos_weight_frame=torch.full((128,), 0.5),
    )
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    for epoch in range(additional_epochs):
        tr, _, avg_grad, steps, scaler = train_one_epoch(
            model,
            cache_loader,
            criterion,
            optimizer,
            scheduler=None,
            device=device,
            use_amp=USE_AMP,
            epoch=epoch,
            scaler=scaler,
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[Continue Training] epoch {epoch + 1}  train={tr:.4f}")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs("model_weights", exist_ok=True)
        torch.save(
            {
                "epoch": additional_epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": tr,
                "config": {
                    "dim": 256,
                    "depth": 6,
                    "num_heads": 8,
                    "learning_rate": 1e-4,
                    "num_epochs": additional_epochs,
                    "n_pitches": 128,
                },
            },
            model_path,
        )
        print(f"Continued model saved to: {model_path}")

def test_model(model_path, num_samples=5, device=None):
    model = load_model_checkpoint(model_path, device)
    model.eval()
    cache_dataset = CashDataset(CACHE_PATH)
    with torch.no_grad():
        for i in range(min(num_samples, len(cache_dataset))):
            x, labels = cache_dataset[i]
            x = x.unsqueeze(0)
            # Ensure input is on the correct device
            x = x.to(device, dtype=torch.float32)
            mark_cudagraph_step_begin()
            out = model(x)
            out = clone_model_outputs(out)
            y = {}
            for k in LABEL_KEYS:
                if k in labels:
                    v = labels[k]
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    # Ensure label is on the correct device
                    y[k] = v.to(device=device, dtype=torch.float32)
            pred_frame = out["frame"][0].detach().cpu()
            true_frame = y["frame"][0].detach().cpu()
            pred_on = out["on"][0].detach().cpu()
            true_on = y["on"][0].detach().cpu()
            pred_off = out["off"][0].detach().cpu()
            true_off = y["off"][0].detach().cpu()
            print(f"Sample {i+1} - True Label:")
            show_on_off_overlay(true_frame, 12, 21, 108)
            print(f"Sample {i+1} - Predicted Frame Distribution:")
            show_on_off_overlay(pred_frame, 12, 21, 108)
            print(f"Sample {i+1} - Predicted Label:")
            show_on_off_predictions(
                pred_on, pred_off, pred_frame, threshold=0.5, S=12, pitch_lo=21, pitch_hi=108
            )

def compute_f1_score(model, dataloader, threshold=0.5, device=None):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x, labels in tqdm(dataloader, desc="Computing F1", unit="batch"):
            # Ensure input is on the correct device
            x = x.to(device, dtype=torch.float32)
            mark_cudagraph_step_begin()
            out = model(x)
            pred_frame = torch.sigmoid(out["frame"]) > threshold
            true_frame = labels["frame"]
            all_true.append(true_frame.cpu().float().flatten())
            all_pred.append(pred_frame.cpu().float().flatten())
    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    tp = (all_true * all_pred).sum().float()
    fp = ((1 - all_true) * all_pred).sum().float()
    fn = (all_true * (1 - all_pred)).sum().float()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1.item()

def run(local_rank, run_name=None, checkpoint_interval=1, amp=False):
    """Top‑level training/experiment entrypoint.  called once per process.

    Args:
        local_rank: process rank for DDP
        run_name: optional name for this run (creates training_runs/<run_name>)
        checkpoint_interval: save checkpoint every N epochs
    """
    # allow modifying the module-level flag from within this function
    global USE_COMPILE

    # apply command-line AMP override before doing anything else
    global USE_AMP
    USE_AMP = bool(amp)

    # set up DDP if necessary
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # CUDA tuning
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, local_rank)

    # create run directory and paths (only rank 0 writes files)
    run_dir = None
    metrics_path = None
    if local_rank == 0:
        run_dir = setup_run_dir(run_name)
        # gather config / env information
        gpu_props = torch.cuda.get_device_properties(local_rank)
        config_dict = {
            "dataset_params": {
                "cache_path": CACHE_PATH,
                "n_mels": N_MELS,
                "beats_per_clip": BEATS_PER_CLIP,
                "subdivisions_per_beat": SUBDIVISIONS_PER_BEAT,
                "hop_length": HOP_LENGTH,
                "sample_rate": SAMPLE_RATE,
            },
            "model_params": {"dim": 256, "depth": 6, "num_heads": 8, "n_pitches": 128},
            "training_params": {
                "batch_size": 4,
                "accumulation_steps": 2,
                "lr": 1e-4,
                "num_epochs": NUM_EPOCHS,
                "scheduler": "onecycle",
            },
            "env": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": gpu_props.name,
                # convert seeds to plain Python ints for JSON dumping
                "torch_seed": int(torch.initial_seed()),
                "numpy_seed": int(np.random.get_state()[1][0]),
                "python_random_state": int(random.getstate()[1][0]),
            },
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        # ensure an empty metrics file exists
        open(metrics_path, "w").close()

    cache_dataset = CashDataset(CACHE_PATH)
    sampler = DistributedSampler(cache_dataset) if world_size > 1 else None
    cache_loader = DataLoader(
        cache_dataset,
        batch_size=4,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    # validation loader (same dataset for simplicity); metrics computed on rank 0
    val_dataset = CashDataset(CACHE_PATH)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = HFTModel(dim=256, num_heads=8, n_pitches=128, n_mels=N_MELS).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    """a
    if world_size > 1:
        # Dummy forward pass to initialize lazy modules before DDP
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 100, N_MELS, device=device)
            _ = model(dummy_input)
        # Move model to device after lazy initialization
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        model = model.to(device)
    """

    if USE_COMPILE:
        # torch.compile ccurrently behaves unpredictably when AMP is enabled
        # (datatype mismatches in conv/batchnorm).  to keep behavior identical
        # to the original `test.py` we simply avoid compiling when `USE_AMP` is
        # true.  the old warning message explains why.
        if USE_AMP:
            if local_rank == 0:
                print("AMP enabled – disabling torch.compile due to known dtype issues")
            USE_COMPILE = False
        else:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                if local_rank == 0:
                    print("Using torch.compile with reduce-overhead mode")
            except Exception as e:
                # Some 2.8 builds reject passing mode and options together; retry options-only.
                if "Either mode or options can be specified" in str(e):
                    try:
                        model = torch.compile(model, options={"triton.cudagraphs": False})
                        if local_rank == 0:
                            print("Using torch.compile with options-only (cudagraphs disabled)")
                    except Exception as e2:
                        if local_rank == 0:
                            print(f"torch.compile failed: {e2}, falling back to eager mode")
                        USE_COMPILE = False
                else:
                    if local_rank == 0:
                        print(f"torch.compile failed: {e}, falling back to eager mode")
                    USE_COMPILE = False
    else:
        if local_rank == 0:
            print("Triton not available, using eager mode")

    opt = make_optimizer(model, lr=1e-4)
    scheduler = make_scheduler(
        opt,
        scheduler_type="onecycle",
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(cache_loader),
        max_lr=1e-4,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    criterion = OnsetsAndFramesPaperLoss(
        lambda_on=1.0,
        lambda_frame=1.0,
        lambda_off=0.5,
        onset_frame_weight=5.0,
        onset_window=4,
    )

    accumulation_steps = 2
    global_step = 0
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()
        tr, lr_track, avg_grad, steps, scaler = train_one_epoch(
            model,
            cache_loader,
            criterion,
            opt,
            scheduler=scheduler,
            device=device,
            use_amp=USE_AMP,
            accumulation_steps=accumulation_steps,
            epoch=epoch,
            scaler=scaler,
        )
        global_step += steps

        # compute validation metrics on rank 0 (own metrics file)
        val_metrics = None
        if local_rank == 0:
            val_metrics = compute_eval_metrics(model, val_loader, criterion, device)
            # optionally save a small batch of predictions
            batch = next(iter(val_loader))
            x_val, y_val = batch
            x_val = x_val.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                mark_cudagraph_step_begin()
                out_val = model(x_val)
                out_val = clone_model_outputs(out_val)
            preds = {k: v.cpu() for k, v in out_val.items()}
            truths = {k: v.cpu() for k, v in y_val.items()}
            torch.save({"preds": preds, "truths": truths},
                       os.path.join(run_dir, "val_preds", f"epoch_{epoch}.pt"))

        epoch_seconds = time.perf_counter() - epoch_start

        if local_rank == 0:
            # print a status line
            print(
                f"[Onset and Frames] epoch {epoch+1}/{NUM_EPOCHS}  "
                f"train={tr:.4f}  vlr_end={opt.param_groups[0]['lr']:.2e}  "
                f"time={epoch_seconds:.2f}s"
            )

            # compose metrics entry
            entry = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": tr,
                "lr": opt.param_groups[0]["lr"],
                "grad_norm": avg_grad,
                "epoch_seconds": epoch_seconds,
            }
            if val_metrics is not None:
                entry.update({
                    "val_loss": val_metrics["loss"],
                    "onset_f1": val_metrics["onset_f1"],
                    "frame_f1": val_metrics["frame_f1"],
                    "onset_prec": val_metrics["onset_prec"],
                    "onset_rec": val_metrics["onset_rec"],
                    "frame_prec": val_metrics["frame_prec"],
                    "frame_rec": val_metrics["frame_rec"],
                })
            log_metrics(metrics_path, entry)

            # checkpointing
            is_best = False
            if val_metrics is not None and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                is_best = True
            if (epoch + 1) % checkpoint_interval == 0 or is_best:
                save_checkpoint(run_dir, epoch + 1, model, opt, scheduler,
                                scaler,
                                global_step, is_best=is_best)

    if local_rank == 0:
        os.makedirs("model_weights", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"model_weights/onset_and_frames_{timestamp}.pth"
        final_state = {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.module.state_dict()
            if world_size > 1
            else model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": tr,
            "config": {
                "dim": 256,
                "depth": 6,
                "num_heads": 8,
                "batch_size": 16,
                "learning_rate": 1e-4,
                "num_epochs": NUM_EPOCHS,
                "n_pitches": 128,
            },
        }
        torch.save(final_state, model_path)
        print(f"Model saved to: {model_path}")
        # also mirror to run directory if available
        if run_dir is not None:
            torch.save(final_state, os.path.join(run_dir, "final_model.pt"))

        # quick evaluation on one batch
        model.eval()
        with torch.no_grad():
            batch = next(iter(cache_loader))
            x, labels = batch
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = {}
            label_tensors = []
            shapes_valid = True
            for k in LABEL_KEYS:
                if k in labels:
                    v = labels[k]
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    label_tensors.append(v)
                    if len(label_tensors) > 1 and v.shape != label_tensors[0].shape:
                        shapes_valid = False
            if label_tensors and shapes_valid:
                label_batch = torch.stack(label_tensors, dim=0).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                y = dict(zip(LABEL_KEYS, label_batch.unbind(0)))
            else:
                if not shapes_valid:
                    print(f"Warning: Label shapes don't match: "
                          f"{[v.shape for v in label_tensors]}")
                for k in LABEL_KEYS:
                    if k in labels:
                        v = labels[k]
                        if v.dim() == 2:
                            v = v.unsqueeze(0)
                        y[k] = v.to(device=device,
                                    dtype=torch.float32,
                                    non_blocking=True)
            mark_cudagraph_step_begin()
            out = model(x)
            out = clone_model_outputs(out)
            pred_frame = out["frame"][0].detach().cpu()
            true_frame = y["frame"][0].detach().cpu()
            show_on_off_overlay(pred_frame, 12, 21, 108)
            show_on_off_overlay(true_frame, 12, 21, 108)

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp", action="store_true",
                        help="enable automatic mixed precision (default off, matches earlier test)")
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional name for this training run")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Save a checkpoint every N epochs")
    args = parser.parse_args()
    run(args.local_rank, run_name=args.run_name,
        checkpoint_interval=args.checkpoint_interval,
        amp=args.amp)

if __name__ == "__main__":
    main()