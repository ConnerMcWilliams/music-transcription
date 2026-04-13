import argparse
import concurrent.futures
import itertools
import json
import os
import re
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pretty_midi import PrettyMIDI
from tqdm.auto import tqdm

from dataset.transforms import midi_notes_to_beat_labels
from utils.beat_utils import (
    get_beats_and_downbeats,
    build_subdivision_times_for_time_range,
)


# ---------------------------------------------------------------------------
# Index building  (frame-based windows, one per piece)
# ---------------------------------------------------------------------------

def build_midi_index(
    list_dir: str,
    d_feature: str,
    d_midi: str,
    num_frame: int,
) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """
    Read train/test/valid .list files and compute non-overlapping frame windows
    for every piece that has both a spectrogram pickle and a MIDI file.

    Returns:
        [(fname, [(start_frame, end_frame), ...]), ...]
    """
    index: List[Tuple[str, List[Tuple[int, int]]]] = []

    for split in ("train", "test", "valid"):
        list_path = os.path.join(list_dir, f"{split}.list")
        if not os.path.exists(list_path):
            continue

        with open(list_path, "r", encoding="utf-8") as fh:
            fnames = [line.rstrip("\n") for line in fh if line.strip()]

        for fname in fnames:
            midi_path = os.path.join(d_midi, fname + ".mid")
            spec_path = os.path.join(d_feature, fname + ".pkl")

            if not (os.path.exists(midi_path) and os.path.exists(spec_path)):
                continue

            # Determine total spec frames (T) without loading full tensor
            try:
                with open(spec_path, "rb") as fh:
                    spec = pickle.load(fh)
                if isinstance(spec, (tuple, list)):
                    spec = spec[0]
                if isinstance(spec, torch.Tensor):
                    T = spec.shape[-1]
                else:
                    T = np.array(spec).shape[-1]
            except Exception as e:
                print(f"  Skipping {fname} -- spec load error: {e}")
                continue

            # Non-overlapping windows of num_frame; last window may be shorter
            windows: List[Tuple[int, int]] = []
            for s in range(0, T, num_frame):
                windows.append((s, min(s + num_frame, T)))

            if windows:
                index.append((fname, windows))

    return index


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def find_checkpoint(cache_dir: str, lookback: int = 100) -> Tuple[int, int]:
    """
    Scan cache_dir for existing midi_*.pkl files to find a safe resume point.
    """
    pattern = re.compile(r"midi_(\d+)\.pkl")
    indices: List[int] = []

    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            m = pattern.match(filename)
            if m:
                indices.append(int(m.group(1)))

    if not indices:
        print("No cached windows found -- starting from beginning.")
        return 0, -1

    max_idx = max(indices)
    start_idx = max(0, max_idx - lookback + 1)
    print(f"Checkpoint: max completed window={max_idx}, resuming from window {start_idx}.")
    return start_idx, max_idx


# ---------------------------------------------------------------------------
# Pickle helper
# ---------------------------------------------------------------------------

def pickle_midi(
    midi_labels: Dict[str, torch.Tensor],
    ts_target: torch.Tensor,
    index: int,
    cache_dir: str,
) -> None:
    """
    Serialize one variable-length beat-normalized MIDI window.

    Files written:
        midi_{index}.pkl       -- {"on","off","frame"} each [128, L]
        ts_target_{index}.pkl  -- [L] float32
    """
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"midi_{index}.pkl"), "wb") as fh:
        pickle.dump(midi_labels, fh, protocol=4)
    with open(os.path.join(cache_dir, f"ts_target_{index}.pkl"), "wb") as fh:
        pickle.dump(ts_target, fh, protocol=4)


# ---------------------------------------------------------------------------
# Module-level worker  (picklable -- required for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_midi_worker(worker_args: tuple) -> Tuple[int, str]:
    """
    Process one piece and all its assigned frame windows.

    Args:
        worker_args: (fname, frame_windows, d_midi, cache_dir, cfg, window_start_idx)

    Returns:
        (num_windows_saved, fname)
    """
    (fname, frame_windows, d_midi, cache_dir, cfg, window_start_idx) = worker_args

    S  = cfg["S"]
    dt = cfg["dt"]

    try:
        pm = PrettyMIDI(os.path.join(d_midi, fname + ".mid"))
        beats, _ = get_beats_and_downbeats(pm)

        windows_saved = 0
        for local_idx, (start_frame, end_frame) in enumerate(frame_windows):
            window_idx = window_start_idx + local_idx
            t_start = start_frame * dt
            t_end = end_frame * dt

            ts_target, max_value, num_beats = build_subdivision_times_for_time_range(
                beats, S, t_start, t_end
            )

            if ts_target is None or len(ts_target) == 0:
                # No complete beat intervals in this window -- write empty
                empty = torch.zeros(128, 0, dtype=torch.float32)
                midi_labels = {"on": empty, "off": empty.clone(), "frame": empty.clone()}
                pickle_midi(midi_labels, torch.zeros(0, dtype=torch.float32), window_idx, cache_dir)
            else:
                on, off, frm = midi_notes_to_beat_labels(
                    pm, ts_target, S, max_value, num_beats
                )
                midi_labels = {
                    "on":    on.contiguous(),    # [128, L]
                    "off":   off.contiguous(),   # [128, L]
                    "frame": frm.contiguous(),   # [128, L]
                }
                pickle_midi(midi_labels, ts_target, window_idx, cache_dir)

            windows_saved += 1

        return (windows_saved, fname)

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return (0, fname)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def transform_midi_and_pickle(
    list_dir: str,
    d_feature: str,
    d_midi: str,
    cache_dir: str,
    *,
    S: int,
    num_frame: int,
    dt: float,
    use_parallel: bool = True,
    num_workers: Optional[int] = None,
) -> None:
    """
    Build variable-length beat-normalized MIDI pickle cache.

    For each non-overlapping spectrogram window of num_frame frames, produces
    a midi_N.pkl with on/off/frame matrices and a ts_target_N.pkl with the
    beat-subdivision times that fall within that window.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Build window index
    print("Building window index...")
    index = build_midi_index(list_dir, d_feature, d_midi, num_frame)
    total_windows = sum(len(wins) for _, wins in index)
    print(f"  {len(index)} pieces  ->  {total_windows} windows total")

    if total_windows == 0:
        print("Nothing to process.")
        return

    # Find checkpoint
    start_window_idx, _ = find_checkpoint(cache_dir)

    # Build filtered worker args
    cfg: dict = {"S": S, "dt": dt}
    worker_args_list: List[tuple] = []
    window_counter = 0

    for fname, wins in index:
        file_window_start = window_counter
        filtered: List[Tuple[int, int]] = []
        for w in wins:
            if window_counter >= start_window_idx:
                filtered.append(w)
            window_counter += 1

        if filtered:
            first_global = file_window_start + (len(wins) - len(filtered))
            worker_args_list.append((
                fname, filtered, d_midi, cache_dir, cfg, first_global,
            ))

    unprocessed = sum(len(a[1]) for a in worker_args_list)
    if unprocessed == 0:
        print("All windows already cached!")
        return

    # Execute
    if use_parallel and len(worker_args_list) > 1:
        if num_workers is None:
            num_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        max_pending = num_workers + 1
        print(f"Parallel caching: {num_workers} workers, {unprocessed} windows remaining")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(
                total=unprocessed, desc="Caching (parallel)",
                unit="window", dynamic_ncols=True,
            ) as pbar:
                pending = {}
                it = iter(worker_args_list)

                for a in itertools.islice(it, max_pending):
                    fut = executor.submit(_process_midi_worker, a)
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
                        a = next(it, None)
                        if a is not None:
                            fut = executor.submit(_process_midi_worker, a)
                            pending[fut] = a
    else:
        print(f"Serial caching: {unprocessed} windows remaining")
        with tqdm(
            total=unprocessed, desc="Caching (serial)",
            unit="window", dynamic_ncols=True,
        ) as pbar:
            for args in worker_args_list:
                try:
                    n, _ = _process_midi_worker(args)
                    pbar.update(n)
                except Exception as e:
                    print(f"Worker error on {args[0]}: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from experiment.config import beat_normalized_spectrogram

    parser = argparse.ArgumentParser(
        description=(
            "Cache variable-length beat-normalized MIDI labels.\n"
            "For each fixed-frame spectrogram window, writes midi_N.pkl\n"
            "(on/off/frame matrices) and ts_target_N.pkl (subdivision times)."
        )
    )
    parser.add_argument("-d_list",      required=True, help="Directory with train/test/valid .list files")
    parser.add_argument("-d_feature",   required=True, help="Pre-computed spectrogram pickles  ({fname}.pkl)")
    parser.add_argument("-d_midi",      required=True, help="MIDI files for beat extraction     ({fname}.mid)")
    parser.add_argument("-d_out_midi",  required=True, help="Output directory for beat-normalized MIDI pickles")
    parser.add_argument("-d_config",    required=True, help="Path to dataset config.json")
    parser.add_argument("-S",           type=int,  default=None, help="Subdivisions per beat  (default: experiment config)")
    parser.add_argument("--serial",     action="store_true",     help="Disable multiprocessing")
    parser.add_argument("--workers",    type=int,  default=None, help="Parallel worker count")
    args = parser.parse_args()

    # Load dataset config.json
    with open(args.d_config, "r", encoding="utf-8") as f:
        dataset_config = json.load(f)

    sr         = dataset_config["feature"]["sr"]
    hop_sample = dataset_config["feature"]["hop_sample"]
    num_frame  = dataset_config["input"]["num_frame"]
    dt         = hop_sample / sr
    S          = args.S or beat_normalized_spectrogram["SUBDIVISIONS_PER_BEAT"]

    print("** cache_spec: variable-length beat-normalized MIDI **")
    print(f"  list dir    : {args.d_list}")
    print(f"  spectrograms: {args.d_feature}")
    print(f"  MIDI        : {args.d_midi}")
    print(f"  output      : {args.d_out_midi}")
    print(f"  S={S}, num_frame={num_frame}, sr={sr}, hop_sample={hop_sample}, dt={dt:.6f}s")

    transform_midi_and_pickle(
        list_dir    = args.d_list,
        d_feature   = args.d_feature,
        d_midi      = args.d_midi,
        cache_dir   = args.d_out_midi,
        S           = S,
        num_frame   = num_frame,
        dt          = dt,
        use_parallel = not args.serial,
        num_workers  = args.workers,
    )

    print("** done **")