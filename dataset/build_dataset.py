"""
build_dataset.py — Pre-pack a corpus split into mmap-able arrays.

Consumes:
    {d_feature}/{fname}.pkl       — per-piece spectrogram, [n_mels, T_full]
                                    (or (tensor, hop_length) tuple)
    {d_midi_cache}/midi_{idx}.pkl — beat-grid {"on","off","frame"} each [128, L]
    {d_midi_cache}/ts_target_{idx}.pkl — [L] beat times (absolute seconds)
    {d_list}/{split}.list         — one fname per line

Produces, per split, in {d_dataset}/{split}/:
    spec.npy          float32 [total_T, n_mels]
    midi_on.npy       float32 [total_x, 128]
    midi_off.npy      float32 [total_x, 128]
    midi_frame.npy    float32 [total_x, 128]
    ts_target.npy     float32 [total_x]   window-relative seconds
    index.npz         int64 arrays spec_start, spec_end, midi_start, midi_end

Window iteration order matches dataset/cache_spec.build_midi_index() so the
global window index used to look up midi_{idx}.pkl stays consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

SPLITS = ("train", "valid", "test")
LABEL_KEYS = ("on", "off", "frame")


def _unwrap(v):
    """Coerce spec/label storage forms to a numpy ndarray (matches refine_dataset._to_tensor)."""
    if isinstance(v, (tuple, list)):
        if len(v) > 0 and isinstance(v[0], (torch.Tensor, np.ndarray)):
            v = v[0]
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v)


def _spec_full_T(spec_path: str) -> Optional[int]:
    try:
        with open(spec_path, "rb") as fh:
            spec = pickle.load(fh)
    except Exception as exc:
        print(f"  Skipping {spec_path}: {exc}")
        return None
    arr = _unwrap(spec)
    return int(arr.shape[-1])


def _midi_L(midi_path: str) -> Optional[int]:
    try:
        with open(midi_path, "rb") as fh:
            d = pickle.load(fh)
    except Exception:
        return None
    on = d["on"]
    arr = _unwrap(on)
    return int(arr.shape[-1])


def _enumerate_windows(
    list_path: str,
    d_feature: str,
    d_midi_cache: str,
    num_frame: int,
    global_idx_start: int,
) -> Tuple[List[dict], int]:
    """
    Walk one split's list file in order. Returns (records, next_global_idx).

    Each record:
        {"fname", "spec_path", "start_frame", "end_frame",
         "global_idx", "midi_path", "ts_path", "L"}

    Skips windows whose midi_{idx}.pkl / ts_target_{idx}.pkl are missing
    (mirrors experiment.refine_experiment.build_metadata).
    """
    if not os.path.exists(list_path):
        return [], global_idx_start

    with open(list_path, "r", encoding="utf-8") as fh:
        fnames = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

    # ── Phase 1a: in-order list of pieces with valid spec, T_full computed in parallel.
    entries: List[Tuple[str, str]] = []
    for fname in fnames:
        spec_path = os.path.join(d_feature, fname + ".pkl")
        if os.path.exists(spec_path):
            entries.append((fname, spec_path))

    def _measure(item):
        fname, spec_path = item
        return fname, spec_path, _spec_full_T(spec_path)

    with ThreadPoolExecutor() as pool:
        measured = list(tqdm(
            pool.map(_measure, entries),
            total=len(entries),
            desc=f"  spec measure ({os.path.basename(list_path)})",
            leave=False,
        ))

    # ── Phase 1b: enumerate windows in order, assign global indices, probe midi cache.
    records: List[dict] = []
    global_idx = global_idx_start

    midi_probe: List[Tuple[int, str, str]] = []  # (record_idx, midi_path, ts_path)

    for fname, spec_path, T_full in measured:
        if T_full is None:
            continue
        for s in range(0, T_full, num_frame):
            e = min(s + num_frame, T_full)
            midi_path = os.path.join(d_midi_cache, f"midi_{global_idx}.pkl")
            ts_path   = os.path.join(d_midi_cache, f"ts_target_{global_idx}.pkl")
            if os.path.exists(midi_path) and os.path.exists(ts_path):
                rec = {
                    "fname":       fname,
                    "spec_path":   spec_path,
                    "start_frame": s,
                    "end_frame":   e,
                    "global_idx":  global_idx,
                    "midi_path":   midi_path,
                    "ts_path":     ts_path,
                    "L":           None,  # filled below
                }
                midi_probe.append((len(records), midi_path, ts_path))
                records.append(rec)
            global_idx += 1

    # ── Phase 1c: read each midi pkl once to get L (parallel).
    def _probe(item):
        rec_idx, midi_path, _ts_path = item
        return rec_idx, _midi_L(midi_path)

    with ThreadPoolExecutor() as pool:
        for rec_idx, L in tqdm(
            pool.map(_probe, midi_probe),
            total=len(midi_probe),
            desc=f"  midi measure ({os.path.basename(list_path)})",
            leave=False,
        ):
            records[rec_idx]["L"] = 0 if L is None else L

    return records, global_idx


def _fill_split(
    split: str,
    records: List[dict],
    out_dir: str,
    n_mels: int,
    label_pitch_dim: int,
    dt: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    total_T = sum(r["end_frame"] - r["start_frame"] for r in records)
    total_x = sum(r["L"] for r in records)
    N = len(records)
    print(f"  {split}: {N} windows  total_T={total_T}  total_x={total_x}")

    spec_arr = np.lib.format.open_memmap(
        os.path.join(out_dir, "spec.npy"),
        mode="w+", dtype=np.float32, shape=(total_T, n_mels),
    )
    on_arr   = np.lib.format.open_memmap(
        os.path.join(out_dir, "midi_on.npy"),
        mode="w+", dtype=np.float32, shape=(total_x, label_pitch_dim),
    )
    off_arr  = np.lib.format.open_memmap(
        os.path.join(out_dir, "midi_off.npy"),
        mode="w+", dtype=np.float32, shape=(total_x, label_pitch_dim),
    )
    frm_arr  = np.lib.format.open_memmap(
        os.path.join(out_dir, "midi_frame.npy"),
        mode="w+", dtype=np.float32, shape=(total_x, label_pitch_dim),
    )
    ts_arr   = np.lib.format.open_memmap(
        os.path.join(out_dir, "ts_target.npy"),
        mode="w+", dtype=np.float32, shape=(total_x,),
    )

    spec_start = np.empty(N, dtype=np.int64)
    spec_end   = np.empty(N, dtype=np.int64)
    midi_start = np.empty(N, dtype=np.int64)
    midi_end   = np.empty(N, dtype=np.int64)

    spec_cursor = 0
    midi_cursor = 0
    cached_spec_path: Optional[str] = None
    cached_spec: Optional[np.ndarray] = None  # [n_mels, T_full]

    for i, rec in enumerate(tqdm(records, desc=f"  fill {split}", unit="win")):
        s, e = rec["start_frame"], rec["end_frame"]
        T_window = e - s

        # Reload the piece's spec only when fname changes.
        if rec["spec_path"] != cached_spec_path:
            with open(rec["spec_path"], "rb") as fh:
                raw = pickle.load(fh)
            arr = _unwrap(raw).astype(np.float32, copy=False)
            if arr.ndim == 3:
                arr = arr[0]  # [1, n_mels, T] -> [n_mels, T]
            cached_spec_path = rec["spec_path"]
            cached_spec = arr

        assert cached_spec is not None
        assert cached_spec.shape[0] == n_mels, (
            f"n_mels mismatch at {rec['fname']}: got {cached_spec.shape[0]}, expected {n_mels}"
        )

        spec_arr[spec_cursor:spec_cursor + T_window] = cached_spec[:, s:e].T
        spec_start[i] = spec_cursor
        spec_end[i]   = spec_cursor + T_window
        spec_cursor  += T_window

        L = rec["L"]
        midi_start[i] = midi_cursor
        midi_end[i]   = midi_cursor + L

        if L > 0:
            with open(rec["midi_path"], "rb") as fh:
                d = pickle.load(fh)
            on  = _unwrap(d["on"]).astype(np.float32, copy=False)     # [128, L]
            off = _unwrap(d["off"]).astype(np.float32, copy=False)
            frm = _unwrap(d["frame"]).astype(np.float32, copy=False)

            with open(rec["ts_path"], "rb") as fh:
                ts_raw = pickle.load(fh)
            ts = _unwrap(ts_raw).astype(np.float32, copy=False)        # [L] absolute seconds

            on_arr [midi_cursor:midi_cursor + L] = on.T
            off_arr[midi_cursor:midi_cursor + L] = off.T
            frm_arr[midi_cursor:midi_cursor + L] = frm.T
            ts_arr [midi_cursor:midi_cursor + L] = ts - (s * dt)

        midi_cursor += L

    # Flush memmaps before writing the index.
    for arr in (spec_arr, on_arr, off_arr, frm_arr, ts_arr):
        arr.flush()
    del spec_arr, on_arr, off_arr, frm_arr, ts_arr

    np.savez(
        os.path.join(out_dir, "index.npz"),
        spec_start=spec_start,
        spec_end=spec_end,
        midi_start=midi_start,
        midi_end=midi_end,
    )


def build(
    d_dataset: str,
    d_list: str,
    d_feature: str,
    d_midi_cache: str,
    num_frame: int,
    n_mels: int,
    dt: float,
    label_pitch_dim: int = 128,
) -> None:
    os.makedirs(d_dataset, exist_ok=True)

    # Global window numbering is monotonically increasing across train/test/valid
    # in the same order as cache_spec.build_midi_index iterates.
    global_idx = 0
    per_split: dict = {}
    for split in SPLITS:
        list_path = os.path.join(d_list, f"{split}.list")
        records, global_idx = _enumerate_windows(
            list_path, d_feature, d_midi_cache, num_frame, global_idx,
        )
        per_split[split] = records
        print(f"  enumerated {split}: {len(records)} windows (global_idx -> {global_idx})")

    for split, records in per_split.items():
        if not records:
            print(f"  {split}: no records, skipping")
            continue
        _fill_split(
            split, records,
            out_dir=os.path.join(d_dataset, split),
            n_mels=n_mels,
            label_pitch_dim=label_pitch_dim,
            dt=dt,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Pack per-window spec slices and beat-grid MIDI labels into "
        "monolithic .npy files for memory-mapped consumption by RefineDataset."
    ))
    parser.add_argument("-d_dataset",    required=True, help="Output root (per-split subdirs created here)")
    parser.add_argument("-d_list",       required=True, help="Dir with train/test/valid .list files")
    parser.add_argument("-d_feature",    required=True, help="Per-piece spectrogram pkl dir ({fname}.pkl)")
    parser.add_argument("-d_midi_cache", required=True, help="cache_spec.py output dir (midi_N.pkl, ts_target_N.pkl)")
    parser.add_argument("-d_config",     required=True, help="dataset/config.json")
    parser.add_argument("--num_frame",   type=int, default=None, help="Override config['input']['num_frame']")
    parser.add_argument("--n_mels",      type=int, default=None, help="Override config['feature']['mel_bins']")

    args = parser.parse_args()

    with open(args.d_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sr         = cfg["feature"]["sr"]
    hop_sample = cfg["feature"]["hop_sample"]
    num_frame  = args.num_frame if args.num_frame is not None else cfg["input"]["num_frame"]
    n_mels     = args.n_mels    if args.n_mels    is not None else cfg["feature"]["mel_bins"]
    dt         = hop_sample / sr

    print("** build_dataset **")
    print(f"  list dir       : {args.d_list}")
    print(f"  feature dir    : {args.d_feature}")
    print(f"  midi cache dir : {args.d_midi_cache}")
    print(f"  output         : {args.d_dataset}")
    print(f"  num_frame={num_frame}  n_mels={n_mels}  dt={dt:.6f}s")

    build(
        d_dataset    = args.d_dataset.rstrip("/").rstrip("\\"),
        d_list       = args.d_list.rstrip("/").rstrip("\\"),
        d_feature    = args.d_feature.rstrip("/").rstrip("\\"),
        d_midi_cache = args.d_midi_cache.rstrip("/").rstrip("\\"),
        num_frame    = num_frame,
        n_mels       = n_mels,
        dt           = dt,
    )

    print("** done **")
