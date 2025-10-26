# transforms.py

import math
from typing import Tuple

import torch
import torchaudio

from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    WINDOW_FN,
    POWER,
    LOG_OFFSET,
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
        self.sample_rate = SAMPLE_RATE
        self.hop_length = HOP_LENGTH
        self.log_offset = LOG_OFFSET

        self.mel_spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            power=POWER,
            center=True,
            pad_mode="reflect",
            window_fn=_get_window_fn(WINDOW_FN),
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
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
    T = mel_time.shape[-1]
    device = mel_time.device

    frame_times = torch.arange(
        T, dtype=torch.float32, device=device
    ) * dt  # [T] seconds

    # clamp targets so searchsorted doesn't go OOB
    ts_target = ts_target.to(device=device, dtype=torch.float32)
    ts_target = ts_target.clamp(max=float(frame_times[-1]))

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
    beats,
    S: int,
    start_beat_idx: int,
    num_beats: int,
    num_pitches: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert PrettyMIDI notes to on/off/frame label matrices on a beat-synchronous grid.

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

    # inner helper: map absolute time -> subdivision index in our window
    def time_to_subidx(t: float):
        # find beat segment k where beats[k] <= t < beats[k+1]
        k = int(torch.searchsorted(torch.tensor(beats), torch.tensor(t), right=True).item()) - 1
        if k < start_beat_idx or k >= start_beat_idx + num_beats:
            return None
        k_local = k - start_beat_idx
        t0, t1 = float(beats[k]), float(beats[k+1])
        if t1 <= t0:
            sub = 0
        else:
            frac = (t - t0) / (t1 - t0)
            sub = int(min(S-1, max(0, math.floor(frac * S))))
        return k_local * S + sub

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            p = int(note.pitch)
            i_on = time_to_subidx(note.start)
            i_off = time_to_subidx(note.end)

            if i_on is not None:
                on[p, i_on] = 1.0
                frm[p, i_on] = 1.0  # frame is active starting at onset

            if i_off is not None:
                off[p, i_off] = 1.0

            if (i_on is not None) and (i_off is not None):
                a, b = sorted((i_on, i_off))
                frm[p, a:b+1] = 1.0
            # else: sustain until we lose it – this is already handled partially above

    return on, off, frm