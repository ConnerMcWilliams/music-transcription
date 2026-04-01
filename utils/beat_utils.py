from typing import Tuple
from pretty_midi import PrettyMIDI
import numpy as np
import torch
import math

def midi_to_time_roll_window(
    pm,
    start_time: float,
    end_time: float,
    fs: float,
    num_pitches: int = 128,
    binary: bool = True,
) -> torch.Tensor:
    """
    Build a piano roll only for [start_time, end_time).

    Returns:
        roll: torch.Tensor of shape [num_pitches, T]
    where
        T = ceil((end_time - start_time) * fs)
    """
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    T = int(math.ceil((end_time - start_time) * fs))
    roll = torch.zeros((num_pitches, T), dtype=torch.float32)

    for inst in pm.instruments:
        if inst.is_drum:
            continue

        for note in inst.notes:
            p = int(note.pitch)
            if not (0 <= p < num_pitches):
                continue

            # Skip notes completely outside the window
            if note.end <= start_time or note.start >= end_time:
                continue

            # Clip note to window
            note_start = max(note.start, start_time)
            note_end = min(note.end, end_time)

            # Map time -> frame indices
            i0 = int(math.floor((note_start - start_time) * fs))
            i1 = int(math.ceil((note_end - start_time) * fs))

            # Clamp to valid frame range
            i0 = max(0, min(i0, T))
            i1 = max(0, min(i1, T))

            if i1 <= i0:
                continue

            if binary:
                roll[p, i0:i1] = 1.0
            else:
                roll[p, i0:i1] = float(note.velocity)

    return roll

def get_beats_and_downbeats(pm: PrettyMIDI):
    """Return numpy arrays of beat times (seconds) and downbeat flags the same length."""
    # pretty_midi can give downbeats via estimated time signature grid
    # If pm.get_downbeats() is empty, we fall back to every N beats as a 'bar' later.
    beats = pm.get_beats()              # shape (K,)
    downbeats = pm.get_downbeats()      # shape (B,)
    return beats, downbeats

def build_subdivision_times(beats_target_loco, S, start_beat_idx, num_beats):
    """
    For a contiguous beat window [start_beat_idx, start_beat_idx+num_beats),
    return an array of target times (seconds) of length num_beats*S at uniform
    subdivisions within each beat (piecewise linear).
    """
    k0 = start_beat_idx
    k1 = min(len(beats_target_loco)-1, k0 + num_beats)  # we need k..k+num_beats (so at least k+1 exists)
    # Guard: ensure we have enough beats to define segments
    if k1 <= k0:
        return None, None  # caller will handle
    seg_beats = beats_target_loco[k0:k1+1]  # length num_beats+1
    out = []
    max_value = None if k1 == len(beats_target_loco) else beats_target_loco[k1]
    for i in range(num_beats):
        t0, t1 = float(seg_beats[i]), float(seg_beats[i+1])
        if t1 <= t0:  # degenerate, skip
            step_times = [t0] * S
        else:
            # S evenly-spaced points in [t0, t1), keep last point exclusive to avoid duplicates, sampled in the center of each beat
            step_times = [t0 + (j + 0.5) * (t1 - t0) / S for j in range(S)]
        out.extend(step_times)
    return torch.tensor(out, dtype=torch.float32), max_value  # [num_beats*S]

if __name__ == "__main__" :
    pm = PrettyMIDI('../test.midi')