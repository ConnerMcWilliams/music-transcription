from typing import Tuple
from pretty_midi import PrettyMIDI
import numpy as np
import torch

def get_beats_and_downbeats(pm: PrettyMIDI):
    """Return numpy arrays of beat times (seconds) and downbeat flags the same length."""
    # pretty_midi can give downbeats via estimated time signature grid
    # If pm.get_downbeats() is empty, we fall back to every N beats as a 'bar' later.
    beats = pm.get_beats()              # shape (K,)
    downbeats = pm.get_downbeats()      # shape (B,)
    return beats, downbeats

def build_subdivision_times(beats, S, start_beat_idx, num_beats):
    """
    For a contiguous beat window [start_beat_idx, start_beat_idx+num_beats),
    return an array of target times (seconds) of length num_beats*S at uniform
    subdivisions within each beat (piecewise linear).
    """
    k0 = start_beat_idx
    k1 = min(len(beats)-1, k0 + num_beats)  # we need k..k+num_beats (so at least k+1 exists)
    # Guard: ensure we have enough beats to define segments
    if k1 <= k0:
        return None  # caller will handle
    seg_beats = beats[k0:k0+num_beats+1]  # length num_beats+1
    out = []
    for i in range(num_beats):
        t0, t1 = float(seg_beats[i]), float(seg_beats[i+1])
        if t1 <= t0:  # degenerate, skip
            step_times = [t0] * S
        else:
            # S evenly-spaced points in [t0, t1), keep last point exclusive to avoid duplicates
            step_times = [t0 + (j + 0.5) * (t1 - t0) / S for j in range(S)]
        out.extend(step_times)
    return torch.tensor(out, dtype=torch.float32)  # [num_beats*S]