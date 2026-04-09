# transforms.py

import math
from typing import Tuple

import torch
import torchaudio

import bisect
import numpy as np

from experiment.config import (
    coarse_spectrogram
)

# ---------- Low-level helpers ----------

def _get_window_fn(name: str):
    """
    Map config.WINDOW_FN (like 'hann') to an actual torch window function.
    """
    name = name.lower()
    if name == "hann" or name == "hanning":
        return torch.hann_window
    elif name == "hamming":
        return torch.hamming_window
    # add more if you experiment
    raise ValueError(f"Unsupported window fn {name}")


class MelTransform:
    """
    Picklable callable that does:
        (waveform, sr) -> (mel_db[n_mels, T], hop_length)

    This replaces the closure-based mel_tx that wasn't picklable.
    """

    def __init__(self):
        # Build the torchaudio transform once and hold it as state
        self.sample_rate = coarse_spectrogram.SAMPLE_RATE
        self.hop_length = coarse_spectrogram.HOP_LENGTH
        self.log_offset = coarse_spectrogram.LOG_OFFSET

        self.mel_spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=coarse_spectrogram.SAMPLE_RATE,
            n_fft=coarse_spectrogram.N_FFT,
            hop_length=coarse_spectrogram.HOP_LENGTH,
            win_length=coarse_spectrogram.WIN_LENGTH,
            power=coarse_spectrogram.POWER,
            center=True,
            pad_mode="reflect",
            window_fn=coarse_spectrogram.WINDOW_FN,
            n_mels=coarse_spectrogram.N_MELS,
            f_min=coarse_spectrogram.F_MIN,
            f_max=coarse_spectrogram.F_MAX,
            mel_scale="slaney",
            norm="slaney",
        )

    def __call__(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        """
        waveform: [1, N] mono audio at SAMPLE_RATE
        sr: the actual sampling rate of this waveform (should == SAMPLE_RATE)

        Returns:
            mel_db: [n_mels, T] float32
            hop_length: int
        """
        if sr != self.sample_rate:
            raise ValueError(
                f"MelTransform expected sr={self.sample_rate}, got {sr}"
            )

        spec = self.mel_spect(waveform)  # [1, n_mels, T]
        if spec.dim() == 3:
            spec = spec[0]               # [n_mels, T]

        mel_db = torch.log(spec + self.log_offset).to(torch.float32)
        return mel_db, self.hop_length

def linear_time_warp_mel(
    mel_time: torch.Tensor,
    ts_target: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    mel_time:   [n_mels, T] time-domain mel spectrogram
    ts_target:  [L] times in seconds we want to sample at (monotone increasing)
    dt:         seconds per mel frame (hop_length / sr)
    returns:
        mel_beat: [n_mels, L] via linear interpolation in time
    """
    # Gets the time domain size
    T = mel_time.shape[-1]
    device = mel_time.device

    # Creates a tensor that has the frame times: [0, dt, dt*2, dt*3, ...]
    frame_times = torch.arange(
        T, dtype=torch.float32, device=device
    ) * dt  # [T] seconds

    # Ensure ts_target is on the same device
    ts_target = ts_target.to(device=device, dtype=torch.float32)
    ts_target = ts_target.clamp(max=float(frame_times[-1]))
    
    # searchsorted (keep on device)
    idx1 = torch.searchsorted(frame_times, ts_target, right=False)
    idx0 = (idx1 - 1).clamp(min=0)
    idx1 = idx1.clamp(max=T - 1)

    t0 = frame_times[idx0]
    t1 = frame_times[idx1]
    denom = (t1 - t0).clamp(min=1e-8)
    alpha = ((ts_target - t0) / denom).unsqueeze(0)  # [1, L]

    x0 = mel_time.index_select(dim=1, index=idx0)
    x1 = mel_time.index_select(dim=1, index=idx1)
    mel_beat = x0 * (1 - alpha) + x1 * alpha
    return mel_beat  # [n_mels, L]


def midi_notes_to_beat_labels(
    pm,
    ts_target, # of length L
    S: int,
    max_value: int,
    num_beats: int,
    num_pitches: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert PrettyMIDI notes to on/off/frame label matrices on a beat-synchronous grid.
    Optimized with vectorized searchsorted operations and proper nearest-neighbor rounding.

    returns:
        on  [num_pitches, L]
        off [num_pitches, L]
        frm [num_pitches, L]
    where L = num_beats * S
    """
    L = num_beats * S
    on  = torch.zeros(num_pitches, L, dtype=torch.float32)
    off = torch.zeros_like(on)
    frm = torch.zeros_like(on)

    # Ensure ts_target is a torch tensor on CPU
    if not isinstance(ts_target, torch.Tensor):
        ts_target = torch.tensor(ts_target, dtype=torch.float32)
    elif ts_target.device.type != 'cpu':
        ts_target = ts_target.cpu()
    
    # Collect all notes from non-drum instruments
    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            p = int(note.pitch)
            if 0 <= p < num_pitches:
                all_notes.append((note.start, note.end, p))
    
    if not all_notes:
        return on, off, frm
    
    # Convert to tensors
    note_starts = torch.tensor([n[0] for n in all_notes], dtype=torch.float32)
    note_ends = torch.tensor([n[1] for n in all_notes], dtype=torch.float32)
    pitches = torch.tensor([n[2] for n in all_notes], dtype=torch.long)
    
    # window boundaries: start is first subdivision time; end should be
    # the time of the first beat beyond the window (max_value) if available
    # otherwise fall back to last subdivision timestamp.
    window_start = float(ts_target[0].item())
    if max_value is not None:
        window_end = float(max_value)
    else:
        # add small epsilon to include final subdivision edge
        window_end = float(ts_target[-1].item()) + 1e-6
    
    # Helper function: find nearest frame index using distance-based rounding
    # preferring earlier frame on ties
    def find_nearest_indices(note_times):
        pos = torch.searchsorted(ts_target, note_times, right=False)
        pos = pos.clamp(min=0, max=len(ts_target) - 1)
        before_idx = (pos - 1).clamp(min=0)
        after_idx = pos.clamp(max=len(ts_target) - 1)
        ts_before = ts_target[before_idx]
        ts_after = ts_target[after_idx]
        dist_before = torch.abs(note_times - ts_before)
        dist_after = torch.abs(note_times - ts_after)
        nearest = torch.where(dist_before <= dist_after, before_idx, after_idx)
        return nearest
    
    # Compute nearest indices for raw times
    idx_starts = find_nearest_indices(note_starts)
    idx_ends = find_nearest_indices(note_ends)
    
    # Determine which notes actually overlap the window
    overlap_mask = (note_ends > window_start) & (note_starts < window_end)
    if not overlap_mask.any():
        return on, off, frm
    
    # Onset and offset masks restrict to events inside window bounds
    on_mask = (note_starts >= window_start) & (note_starts < window_end)
    off_mask = (note_ends >= window_start) & (note_ends < window_end)
    
    # Filter by max_value if provided (same as before but apply to masks too)
    if max_value is not None:
        max_mask = note_starts <= (max_value + window_end) / 2
        overlap_mask &= max_mask
        on_mask &= max_mask
        off_mask &= max_mask
        # we don't actually need to trim all_notes here anymore
    
    # Batch set on/off labels using masks
    if on_mask.any():
        on[pitches[on_mask], idx_starts[on_mask]] = 1.0
    if off_mask.any():
        off[pitches[off_mask], idx_ends[off_mask]] = 1.0
    
    # Frame regions: mark only within window overlap
    # clamp note times to window boundaries for frame computation
    starts_clamped = torch.clamp(note_starts, min=window_start, max=window_end)
    ends_clamped = torch.clamp(note_ends, min=window_start, max=window_end)
    idx_start_clamped = find_nearest_indices(starts_clamped)
    idx_end_clamped = find_nearest_indices(ends_clamped)
    
    for i in range(len(note_starts)):
        if not overlap_mask[i]:
            continue
        p = pitches[i].item()
        a = int(min(idx_start_clamped[i].item(), idx_end_clamped[i].item()))
        b = int(max(idx_start_clamped[i].item(), idx_end_clamped[i].item()))
        frm[p, a : b + 1] = 1.0

    return on, off, frm