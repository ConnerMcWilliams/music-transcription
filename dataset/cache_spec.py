#! python
"""
cache_spec.py — Beat-normalize pre-cached spectrograms and labels.

Input (per piece, all keyed by the same fname from .list files):
  {d_feature}/{fname}.pkl  — [n_mels, T] or [1, n_mels, T] float32 mel spectrogram
                             (produced by conv_wav2fe.py / MelTransform)
  {d_label}/{fname}.pkl    — dict {"mpe":      [T, 128] int/float  (0 or 1),
                                   "onset":    [T, 128] float32    (0.0-1.0),
                                   "offset":   [T, 128] float32    (0.0-1.0),
                                   "velocity": [T, 128] int/float  (0-127)}
                             (produced by note2label.py; values may be Python lists)
  {d_midi}/{fname}.mid     — MIDI file  (used ONLY for beat-time extraction)

Output (per window, compatible with CashDataset):
  {d_out}/spectrogram_{idx}.pkl — [1, n_mels, L] float32
  {d_out}/labels_{idx}.pkl      — dict {"on":       [L, 128] float32,
                                        "off":      [L, 128] float32,
                                        "frame":    [L, 128] float32,
                                        "velocity": [L, 128] float32}
  {d_out}/extra_{idx}.pkl       — dict {"times": ts_target [L], "max_time": float|None}

where L = beats_per_window * subdivisions_per_beat.

Label warping strategy:
  "onset"  / "offset"  (soft 0.0-1.0)  ->  linear interpolation
  "mpe"                (binary 0/1)    ->  nearest-neighbour
  "velocity"           (int 0-127)     ->  nearest-neighbour
"""

import argparse
import concurrent.futures
import itertools
import os
import re
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from pretty_midi import PrettyMIDI
from tqdm.auto import tqdm

# Ensure project root is on sys.path so `dataset.*` / `utils.*` resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.transforms import linear_time_warp_mel
from utils.beat_utils import get_beats_and_downbeats, build_subdivision_times


# ---------------------------------------------------------------------------
# Core warping -- operates entirely on pre-computed tensors, no MIDI needed
# ---------------------------------------------------------------------------

def warp_labels_to_beat_grid(
    labels_raw: Dict[str, torch.Tensor],
    ts_target: torch.Tensor,
    dt: float,
) -> Dict[str, torch.Tensor]:
    """
    Warp time-domain labels to a beat-synchronous grid via interpolation.

    Args:
        labels_raw:  {"mpe", "onset", "offset", "velocity"} each [T, 128]
        ts_target:   [L] beat-grid times in seconds  (from build_subdivision_times)
        dt:          seconds per spectrogram frame  = hop_length / sample_rate

    Returns:
        {"on", "off", "frame", "velocity"} each [L, 128]

    Strategy:
        "onset"   / "offset"  -- linear interpolation   (soft float values)
        "mpe"     / "velocity"-- nearest-neighbour      (binary / integer values)
    """
    onset_raw    = labels_raw["onset"]      # [T, 128]
    offset_raw   = labels_raw["offset"]     # [T, 128]
    mpe_raw      = labels_raw["mpe"]        # [T, 128]
    velocity_raw = labels_raw["velocity"]   # [T, 128]

    T      = onset_raw.shape[0]
    device = onset_raw.device

    # Build frame-time axis  [T]  seconds
    frame_times = torch.arange(T, dtype=torch.float32, device=device) * dt
    ts = ts_target.to(device=device, dtype=torch.float32).clamp(max=float(frame_times[-1]))

    # Bracketing indices for every query time
    idx1 = torch.searchsorted(frame_times, ts, right=False)  # [L]
    idx0 = (idx1 - 1).clamp(min=0)
    idx1 = idx1.clamp(max=T - 1)

    t0 = frame_times[idx0]   # [L]
    t1 = frame_times[idx1]   # [L]

    # Nearest-neighbour index: choose closer frame; prefer earlier on tie
    closer = torch.where((ts - t0) <= (t1 - ts), idx0, idx1)  # [L]

    # Linear interpolation weight  [L]
    alpha = ((ts - t0) / (t1 - t0).clamp(min=1e-8))

    def _linear(x: torch.Tensor) -> torch.Tensor:
        # x: [T, 128]  ->  transpose, gather, interpolate, transpose back -> [L, 128]
        xt = x.T.float()                          # [128, T]
        x0 = xt.index_select(1, idx0)             # [128, L]
        x1 = xt.index_select(1, idx1)             # [128, L]
        return (x0 * (1.0 - alpha) + x1 * alpha).T.contiguous()  # [L, 128]

    def _nearest(x: torch.Tensor) -> torch.Tensor:
        # x: [T, 128]  ->  nearest-neighbour gather -> [L, 128]
        return x.T.index_select(1, closer).T.contiguous()

    return {
        "on":       _linear(onset_raw).to(torch.float32),
        "off":      _linear(offset_raw).to(torch.float32),
        "frame":    _nearest(mpe_raw).to(torch.float32),
        "velocity": _nearest(velocity_raw).to(torch.float32),
    }


# ---------------------------------------------------------------------------
# Single-piece generator  (main reusable API)
# ---------------------------------------------------------------------------

def normalize_cached_sample(
    spec: torch.Tensor,
    labels_raw: Dict[str, torch.Tensor],
    midi_path: str,
    *,
    S: int,
    beats_per_window: int,
    hop_beats: int,
    dt: float,
) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Optional[float]]]:
    """
    Generator yielding beat-normalised windows for a single piece.

    Args:
        spec:             [n_mels, T] float32 mel spectrogram
        labels_raw:       {"mpe","onset","offset","velocity"} each [T, 128]
        midi_path:        Path to .mid file -- used ONLY for beat extraction
        S:                Subdivisions per beat
        beats_per_window: Number of beats per output window
        hop_beats:        Stride in beats between consecutive windows
        dt:               hop_length / sample_rate  (seconds per frame)

    Yields:
        norm_spec:    [1, n_mels, L]  float32
        norm_labels:  {"on","off","frame","velocity"} each [L, 128]
        ts_target:    [L]  float32  beat-grid times in seconds
        max_value:    float or None  (end time of window)
    """
    pm = PrettyMIDI(midi_path)
    beats, _ = get_beats_and_downbeats(pm)
    K = max(0, len(beats) - 1)
    if K < beats_per_window:
        return  # piece too short -- skip silently

    # Window starts -- mirrors probe_and_build_index in cache_norm.py
    starts = list(range(0, K - beats_per_window + 1, hop_beats))
    last_start = K - beats_per_window
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    for start_beat_idx in starts:
        ts_target, max_value = build_subdivision_times(beats, S, start_beat_idx, beats_per_window)
        if ts_target is None:
            continue  # degenerate beat window

        norm_spec   = linear_time_warp_mel(spec, ts_target, dt)           # [n_mels, L]
        norm_labels = warp_labels_to_beat_grid(labels_raw, ts_target, dt)  # [L, 128] each

        yield (
            norm_spec.to(torch.float32).unsqueeze(0).contiguous(),  # [1, n_mels, L]
            norm_labels,
            ts_target,
            max_value,
        )


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_spec_index(
    list_dir: str,
    d_feature: str,
    d_label: str,
    d_midi: str,
    beats_per_window: int,
    hop_beats: int,
) -> List[Tuple[str, List[int]]]:
    """
    Read train/test/valid .list files and compute valid beat-window start
    indices for each piece that has all three required input files.

    Returns:
        [(fname, [start_beat_idx, ...]), ...]
        Pieces shorter than beats_per_window are omitted.
    """
    index: List[Tuple[str, List[int]]] = []

    for split in ("train", "test", "valid"):
        list_path = os.path.join(list_dir, f"{split}.list")
        if not os.path.exists(list_path):
            continue

        with open(list_path, "r", encoding="utf-8") as fh:
            fnames = [line.rstrip("\n") for line in fh if line.strip()]

        for fname in fnames:
            midi_path  = os.path.join(d_midi,    fname + ".mid")
            spec_path  = os.path.join(d_feature, fname + ".pkl")
            label_path = os.path.join(d_label,   fname + ".pkl")

            if not (os.path.exists(midi_path) and
                    os.path.exists(spec_path) and
                    os.path.exists(label_path)):
                continue

            try:
                pm = PrettyMIDI(midi_path)
            except Exception as e:
                print(f"  Skipping {fname} -- MIDI load error: {e}")
                continue

            beats, _ = get_beats_and_downbeats(pm)
            K = max(0, len(beats) - 1)
            if K < beats_per_window:
                continue

            starts = list(range(0, K - beats_per_window + 1, hop_beats))
            last_start = K - beats_per_window
            if not starts or starts[-1] != last_start:
                starts.append(last_start)

            index.append((fname, starts))

    return index


# ---------------------------------------------------------------------------
# Checkpoint  (identical logic to cache_norm.find_checkpoint)
# ---------------------------------------------------------------------------

def find_checkpoint(cache_dir: str, lookback: int = 100) -> Tuple[int, int]:
    """
    Scan cache_dir for existing spectrogram_*.pkl files to find a safe resume point.

    Returns:
        (start_window_idx, max_completed_idx)
        max_completed_idx == -1 when nothing has been cached yet.
    """
    pattern = re.compile(r"spectrogram_(\d+)\.pkl")
    indices: List[int] = []

    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            m = pattern.match(filename)
            if m:
                indices.append(int(m.group(1)))

    if not indices:
        print("No cached windows found -- starting from beginning.")
        return 0, -1

    max_idx   = max(indices)
    start_idx = max(0, max_idx - lookback + 1)
    print(f"Checkpoint: max completed window={max_idx}, resuming from window {start_idx}.")
    return start_idx, max_idx


# ---------------------------------------------------------------------------
# Pickle helper  (mirrors cache_norm.pickle_spectrogram)
# ---------------------------------------------------------------------------

def pickle_spectrogram(
    mel: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    index: int,
    cache_dir: str,
    extra: Optional[dict] = None,
) -> None:
    """
    Serialize one beat-normalized (spectrogram, labels[, extra]) triplet.

    Files written:
        spectrogram_{index}.pkl  -- [1, n_mels, L]
        labels_{index}.pkl       -- {"on","off","frame","velocity"} each [L, 128]
        extra_{index}.pkl        -- {"times": ts_target, "max_time": ...}  (optional)
    """
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"spectrogram_{index}.pkl"), "wb") as fh:
        pickle.dump(mel, fh, protocol=4)
    with open(os.path.join(cache_dir, f"labels_{index}.pkl"), "wb") as fh:
        pickle.dump(labels, fh, protocol=4)
    if extra:
        with open(os.path.join(cache_dir, f"extra_{index}.pkl"), "wb") as fh:
            pickle.dump(extra, fh, protocol=4)


# ---------------------------------------------------------------------------
# Module-level worker  (picklable -- required for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_spec_worker(worker_args: tuple) -> Tuple[int, str]:
    """
    Process one piece and all its assigned beat windows.

    Args:
        worker_args: (fname, start_beat_indices, d_feature, d_label, d_midi,
                      cache_dir, config, window_start_idx)

    Returns:
        (num_windows_saved, fname)
    """
    (fname, start_beat_indices, d_feature, d_label,
     d_midi, cache_dir, cfg, window_start_idx) = worker_args

    S                = cfg["S"]
    beats_per_window = cfg["beats_per_window"]
    dt               = cfg["dt"]

    try:
        # Load spectrogram
        with open(os.path.join(d_feature, fname + ".pkl"), "rb") as fh:
            spec = pickle.load(fh)

        # conv_wav2fe.py saves (mel_db, hop_length) tuple
        if isinstance(spec, (tuple, list)):
            spec = spec[0]
        if not isinstance(spec, torch.Tensor):
            spec = torch.as_tensor(np.array(spec), dtype=torch.float32)
        if spec.dim() == 3:
            spec = spec.squeeze(0)   # [n_mels, T]
        spec = spec.to(torch.float32)

        # Load labels
        with open(os.path.join(d_label, fname + ".pkl"), "rb") as fh:
            labels_raw = pickle.load(fh)

        for key in ("mpe", "onset", "offset", "velocity"):
            if key not in labels_raw:
                raise KeyError(f"Label dict missing key '{key}' in {fname}")

        def _to_tensor(v) -> torch.Tensor:
            if isinstance(v, torch.Tensor):
                return v.to(torch.float32)
            arr = np.array(v) if not isinstance(v, np.ndarray) else v
            return torch.as_tensor(arr, dtype=torch.float32)

        labels_raw = {k: _to_tensor(v) for k, v in labels_raw.items()}

        # Load MIDI for beat times only
        pm = PrettyMIDI(os.path.join(d_midi, fname + ".mid"))
        beats, _ = get_beats_and_downbeats(pm)

        # Process each window
        windows_saved = 0
        for local_idx, start_beat_idx in enumerate(start_beat_indices):
            window_idx = window_start_idx + local_idx

            ts_target, max_value = build_subdivision_times(
                beats, S, start_beat_idx, beats_per_window
            )
            if ts_target is None:
                continue

            norm_spec   = linear_time_warp_mel(spec, ts_target, dt)          # [n_mels, L]
            norm_labels = warp_labels_to_beat_grid(labels_raw, ts_target, dt) # [L, 128] each

            mel   = norm_spec.to(torch.float32).unsqueeze(0).contiguous()    # [1, n_mels, L]
            extra = {"times": ts_target, "max_time": max_value}

            pickle_spectrogram(mel, norm_labels, window_idx, cache_dir, extra)
            windows_saved += 1

        return (windows_saved, fname)

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return (0, fname)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def transform_spec_and_pickle(
    list_dir: str,
    d_feature: str,
    d_label: str,
    d_midi: str,
    cache_dir: str,
    *,
    S: int,
    beats_per_window: int,
    hop_beats: int,
    dt: float,
    use_parallel: bool = True,
    num_workers: Optional[int] = None,
) -> None:
    """
    Build a beat-normalized pickle cache from pre-computed spectrograms + labels.

    Mirrors cache_norm.transform_and_pickle() but sources spectrogram and label
    pickles instead of raw audio / MIDI.  Checkpoint / resume logic is identical
    to cache_norm so that partial runs can be safely restarted.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Build window index
    print("Building window index...")
    index = build_spec_index(
        list_dir, d_feature, d_label, d_midi, beats_per_window, hop_beats
    )
    total_windows = sum(len(starts) for _, starts in index)
    print(f"  {len(index)} pieces  ->  {total_windows} windows total")

    if total_windows == 0:
        print("Nothing to process.")
        return

    # Find checkpoint
    start_window_idx, _ = find_checkpoint(cache_dir)

    # Build filtered worker args
    cfg: dict = {"S": S, "beats_per_window": beats_per_window, "dt": dt}
    worker_args_list: List[tuple] = []
    window_counter = 0

    for fname, starts in index:
        file_window_start = window_counter
        filtered: List[int] = []
        for s in starts:
            if window_counter >= start_window_idx:
                filtered.append(s)
            window_counter += 1

        if filtered:
            # global index of the first window in this file's filtered list
            first_global = file_window_start + (len(starts) - len(filtered))
            worker_args_list.append((
                fname, filtered, d_feature, d_label, d_midi,
                cache_dir, cfg, first_global,
            ))

    unprocessed = sum(len(a[1]) for a in worker_args_list)
    if unprocessed == 0:
        print("All windows already cached!")
        return

    # Execute
    if use_parallel and len(worker_args_list) > 1:
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 2) - 1)
        # Limit in-flight futures to avoid OOM from too many loaded spectrograms
        max_pending = num_workers * 2
        print(f"Parallel caching: {num_workers} workers, {unprocessed} windows remaining")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(
                total=unprocessed, desc="Caching (parallel)",
                unit="window", dynamic_ncols=True,
            ) as pbar:
                pending = {}
                it = iter(worker_args_list)

                # Seed initial batch
                for a in itertools.islice(it, max_pending):
                    fut = executor.submit(_process_spec_worker, a)
                    pending[fut] = a

                while pending:
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        del pending[future]
                        try:
                            n, _ = future.result()
                            pbar.update(n)
                        except Exception as e:
                            print(f"Worker error: {e}")
                        # Submit one more to replace the completed one
                        a = next(it, None)
                        if a is not None:
                            fut = executor.submit(_process_spec_worker, a)
                            pending[fut] = a
    else:
        print(f"Serial caching: {unprocessed} windows remaining")
        with tqdm(
            total=unprocessed, desc="Caching (serial)",
            unit="window", dynamic_ncols=True,
        ) as pbar:
            for args in worker_args_list:
                try:
                    n, _ = _process_spec_worker(args)
                    pbar.update(n)
                except Exception as e:
                    print(f"Worker error on {args[0]}: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from experiment.config import coarse_spectrogram, beat_normalized_spectrogram

    parser = argparse.ArgumentParser(
        description=(
            "Beat-normalize pre-cached spectrograms and labels.\n"
            "Reads {d_feature}/{fname}.pkl and {d_label}/{fname}.pkl;\n"
            "writes spectrogram_N.pkl / labels_N.pkl / extra_N.pkl to d_out."
        )
    )
    parser.add_argument("-d_list",      required=True, help="Directory with train/test/valid .list files")
    parser.add_argument("-d_feature",   required=True, help="Pre-computed spectrogram pickles  ({fname}.pkl)")
    parser.add_argument("-d_label",     required=True, help="Pre-computed label pickles         ({fname}.pkl)")
    parser.add_argument("-d_midi",      required=True, help="MIDI files for beat extraction     ({fname}.mid)")
    parser.add_argument("-d_out",       required=True, help="Output directory for beat-normalized pickles")
    parser.add_argument("-S",           type=int,  default=None, help="Subdivisions per beat        (default: config)")
    parser.add_argument("-B",           type=int,  default=None, help="Beats per window             (default: config)")
    parser.add_argument("-hop_beats",   type=int,  default=None, help="Hop in beats between windows (default: B)")
    parser.add_argument("--sr",         type=int,  default=None, help="Spectrogram sample rate      (default: config)")
    parser.add_argument("--hop_length", type=int,  default=None, help="Spectrogram hop length       (default: config)")
    parser.add_argument("--serial",     action="store_true",     help="Disable multiprocessing")
    parser.add_argument("--workers",    type=int,  default=None, help="Parallel worker count")
    args = parser.parse_args()

    # Resolve values -- CLI flags override config defaults
    S          = args.S          or beat_normalized_spectrogram["SUBDIVISIONS_PER_BEAT"]
    B          = args.B          or beat_normalized_spectrogram["BEATS_PER_CLIP"]
    hop_beats  = args.hop_beats  or B
    sr         = args.sr         or coarse_spectrogram["SAMPLE_RATE"]
    hop_length = args.hop_length or coarse_spectrogram["HOP_LENGTH"]
    dt         = hop_length / sr

    print("** cache_spec: beat-normalize cached spectrograms + labels **")
    print(f"  list dir    : {args.d_list}")
    print(f"  spectrograms: {args.d_feature}")
    print(f"  labels      : {args.d_label}")
    print(f"  MIDI        : {args.d_midi}")
    print(f"  output      : {args.d_out}")
    print(f"  S={S}, B={B}, hop_beats={hop_beats}, sr={sr}, hop_length={hop_length}, dt={dt:.6f}s")

    transform_spec_and_pickle(
        list_dir         = args.d_list,
        d_feature        = args.d_feature,
        d_label          = args.d_label,
        d_midi           = args.d_midi,
        cache_dir        = args.d_out,
        S                = S,
        beats_per_window = B,
        hop_beats        = hop_beats,
        dt               = dt,
        use_parallel     = not args.serial,
        num_workers      = args.workers,
    )

    print("** done **")