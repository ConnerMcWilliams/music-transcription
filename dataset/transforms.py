import torch
import torchaudio
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH,WIN_LENGTH, N_MELS,F_MIN,F_MAX,WINDOW_FN,POWER,LOG_OFFSET
from typing import Tuple
import math

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    power=POWER,                  
    center=True,
    pad_mode="reflect",
    window_fn=torch.hann_window,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    mel_scale="slaney",        
    norm="slaney"               
)

def log_mel(waveform):
    S = mel(waveform)          # [1, n_mels, T]
    return torch.log(S + LOG_OFFSET).squeeze(0)  # or torch.log1p(S) if you prefer

def linear_time_warp_mel(mel_time: torch.Tensor, ts_target: torch.Tensor, dt: float) -> torch.Tensor:
    """
    mel_time: [n_mels, T] time-domain mel spectrogram
    ts_target: [L] desired times in seconds to sample (monotone increasing)
    Returns mel_beat: [n_mels, L] via linear interpolation along time.
    """
    # time of each mel frame center (seconds)
    T = mel_time.shape[-1]
    frame_times = torch.arange(T, dtype=torch.float32, device=mel_time.device) * (dt)

    # For each target time, find bracketing indices
    idx1 = torch.searchsorted(frame_times, ts_target.clamp(max=float(frame_times[-1])), right=False)
    idx0 = (idx1 - 1).clamp(min=0)
    idx1 = idx1.clamp(max=T-1)

    t0 = frame_times[idx0]
    t1 = frame_times[idx1]
    denom = (t1 - t0).clamp(min=1e-8)
    alpha = ((ts_target - t0) / denom).unsqueeze(0)  # [1,L]

    x0 = mel_time.index_select(dim=1, index=idx0)
    x1 = mel_time.index_select(dim=1, index=idx1)
    return x0 * (1 - alpha) + x1 * alpha  # [n_mels, L]

def midi_notes_to_beat_labels(pm, beats, 
                              S: int, 
                              start_beat_idx: int, 
                              num_beats: int, 
                              num_pitches: int=128
                              ) -> Tuple[torch.Tensor, ...]:
    """Pure conversion of note times to beat-grid on/off/frame tensors."""
    L = num_beats * S
    on = torch.zeros(num_pitches, L, dtype=torch.float32)
    off = torch.zeros_like(on)
    frm = torch.zeros_like(on)

    # Convenience: map a time (sec) -> (global subdivision index)
    def time_to_subidx(t):
        # find beat segment k s.t. beats[k] <= t < beats[k+1]
        k = int(torch.searchsorted(torch.tensor(beats), torch.tensor(t), right=True).item()) - 1
        # constrain to our window
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

    # Fill labels
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                p = int(note.pitch)
                i_on = time_to_subidx(note.start)
                i_off = time_to_subidx(note.end)
                if i_on is not None:
                    on[p, i_on] = 1.0
                if i_off is not None:
                    off[p, i_off] = 1.0
                # Frames: mark all subs between on/off (inclusive of on, exclusive of off)
                if (i_on is not None) and (i_off is not None):
                    a, b = sorted((i_on, i_off))
                    frm[p, a:b+1] = 1.0
                elif i_on is not None:
                    frm[p, i_on] = 1.0  # at least the onset sub
    return on, off, frm