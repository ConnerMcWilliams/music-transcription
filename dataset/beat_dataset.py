# dataset_beats.py
import os
import numpy as np
from typing import Optional, Callable, Tuple, Dict, Any

import torch
import torchaudio
from torch.utils.data import Dataset
from pretty_midi import PrettyMIDI

from dataset.transforms import linear_time_warp_mel, midi_notes_to_beat_labels
from utils.beat_utils import get_beats_and_downbeats, build_subdivision_times


class MaestroDatasetWithWindowingInBeats(Dataset):
    """
    Returns ONE window per __getitem__:

    If use_beat_warp=True (tempo-normalized):
        mel  -> [1, n_mels, L] where L = beats_per_window * subdivisions
        labels['on'/'off'/'frame'] -> [L, 128]

    If use_beat_warp=False (plain time):
        mel  -> [1, n_mels, T_fixed]
        labels aligned approximately in time, not warped to beats

    meta: { 'sr', 'hop_length', 'beats', 'subs_per_beat', 'start_beat_idx' }
    """

    def __init__(self,
                 metadata,                      # pandas.DataFrame with columns: audio_filename, midi_filename
                 root_dir: str,
                 mel_tx: Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]],
                 *,
                 n_mels: int,
                 subdivisions: int,
                 beats_per_window: int,
                 hop_beats: Optional[int] = None,
                 hop_length: int,
                 sr_target: Optional[int] = None,
                 waveform_tx: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 mel_aug_prewarp: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 mel_aug_postwarp: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 return_time_labels: bool = False,
                 time_roll_rate: Optional[float] = None,
                 use_beat_warp: bool = True,   ### CHANGED: new flag
                 ):
        # --- config ---
        self.metadata = metadata.reset_index(drop=True)
        self.root_dir = root_dir
        self.mel_tx = mel_tx
        self.n_mels = int(n_mels)
        self.S = int(subdivisions)
        self.beats_per_window = int(beats_per_window)
        self.hop_beats = int(hop_beats) if hop_beats is not None else self.beats_per_window
        self.hop_length = hop_length

        self.sr_target = sr_target
        self.waveform_tx = waveform_tx
        self.mel_aug_prewarp = mel_aug_prewarp
        self.mel_aug_postwarp = mel_aug_postwarp
        self.target_transform = target_transform

        self.return_time_labels = return_time_labels
        self.time_roll_rate = float(time_roll_rate) if time_roll_rate is not None else None

        self.use_beat_warp = use_beat_warp   ### CHANGED

        # seconds per mel frame (for warping math); assume hop_length / sr_target
        if sr_target is None or hop_length is None:
            raise ValueError("sr_target and hop_length must be provided")
        self.dt = self.hop_length / self.sr_target

        # build index of (row_idx, start_beat_idx)
        self.index = []
        self._probe_and_build_index()

    def _probe_and_build_index(self) -> None:
        idx = []
        for i, row in self.metadata.iterrows():
            midi_path = os.path.join(self.root_dir, row["midi_filename"])
            try:
                pm = PrettyMIDI(midi_path)
            except Exception:
                continue

            beats, _ = get_beats_and_downbeats(pm)
            K = max(0, len(beats) - 1)
            if K < self.beats_per_window:
                continue

            starts = list(range(0, K - self.beats_per_window + 1, self.hop_beats))
            last_start = K - self.beats_per_window
            if not starts or starts[-1] != last_start:
                starts.append(last_start)

            for s in starts:
                idx.append((i, s))

        # dedupe
        self.index = list(dict.fromkeys(idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        row_idx, start_beat_idx = self.index[idx]
        row = self.metadata.iloc[row_idx]
        audio_path = os.path.join(self.root_dir, row["audio_filename"])
        midi_path  = os.path.join(self.root_dir, row["midi_filename"])

        pm = PrettyMIDI(midi_path)
        beats, _ = get_beats_and_downbeats(pm)

        num_beats = self.beats_per_window
        L = num_beats * self.S  # tokens in beat-warp space

        # ---- audio ----
        waveform, sr = torchaudio.load(audio_path)  # [C, N]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.sr_target is not None and self.sr_target != sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr_target)
            sr = self.sr_target
        if self.waveform_tx is not None:
            waveform = self.waveform_tx(waveform)

        # mel spectrogram in time
        mel_time, hop_len_check = self.mel_tx(waveform, sr)  ### CHANGED: expect (mel, hop_len)
        if mel_time.shape[0] != self.n_mels:
            raise ValueError(f"mel_tx returned {mel_time.shape[0]} mels, expected {self.n_mels}")
        if hop_len_check != self.hop_length:
            # optional sanity check
            pass

        if self.mel_aug_prewarp is not None:
            mel_time = self.mel_aug_prewarp(mel_time)

        if self.use_beat_warp:
            # ----- tempo-normalized branch -----
            ts_target = build_subdivision_times(
                beats, self.S, start_beat_idx, num_beats
            )  # [L] seconds
            if ts_target is None:
                mel = torch.zeros(1, self.n_mels, L, dtype=torch.float32)
                zeros = torch.zeros(L, 128, dtype=torch.float32)
                labels = {"on": zeros.clone(), "off": zeros.clone(), "frame": zeros.clone()}
                meta = {
                    "sr": sr,
                    "hop_length": self.hop_length,
                    "beats": None,
                    "subs_per_beat": self.S,
                    "start_beat_idx": start_beat_idx,
                }
                return mel, labels, meta

            mel_beat = linear_time_warp_mel(mel_time, ts_target, self.dt)  # [n_mels, L]

            if mel_beat.shape != (self.n_mels, L):
                raise RuntimeError(
                    f"mel_beat has shape {tuple(mel_beat.shape)}, expected ({self.n_mels}, {L})"
                )
            mel_for_model = mel_beat  # [n_mels, L]
        else:
            # ----- plain-time branch (no beat warp) -----
            # naive crop: take a window of length L from mel_time based on start_beat_idx
            # convert beat idx → approx frame idx using beats[] time stamps
            start_t = beats[start_beat_idx]
            end_t   = beats[min(start_beat_idx + num_beats, len(beats)-1)]
            # duration in seconds of this window:
            dur = end_t - start_t
            # number of mel frames that span similar duration
            frames_per_sec = sr / self.hop_length
            approx_len = int(round(dur * frames_per_sec * self.S / (self.S / 1)))  
            # fall back to L if weird
            Tcrop = max(L, approx_len)
            t0_frames = int(round(start_t * frames_per_sec))
            t1_frames = t0_frames + Tcrop
            mel_slice = mel_time[:, t0_frames:t1_frames]

            # pad/truncate to L time steps for consistency
            if mel_slice.shape[1] < L:
                pad_amt = L - mel_slice.shape[1]
                mel_slice = torch.nn.functional.pad(mel_slice, (0, pad_amt))
            elif mel_slice.shape[1] > L:
                mel_slice = mel_slice[:, :L]

            mel_for_model = mel_slice  # [n_mels, L]

        if self.mel_aug_postwarp is not None:
            mel_for_model = self.mel_aug_postwarp(mel_for_model)

        # ---- labels (always beat-aligned right now) ----
        on_b, off_b, frm_b = midi_notes_to_beat_labels(
            pm, beats, self.S, start_beat_idx, num_beats
        )  # [128, L] or [L,128]

        on_t  = torch.as_tensor(on_b,  dtype=torch.float32)
        off_t = torch.as_tensor(off_b, dtype=torch.float32)
        frm_t = torch.as_tensor(frm_b, dtype=torch.float32)

        if on_t.shape == (128, L):  # transpose to [L,128]
            on_t  = on_t.transpose(0, 1)
            off_t = off_t.transpose(0, 1)
            frm_t = frm_t.transpose(0, 1)
        elif on_t.shape != (L, 128):
            raise RuntimeError(f"Expected targets [L,128], got {tuple(on_t.shape)}")

        if self.target_transform is not None:
            on_t  = self.target_transform(on_t)
            off_t = self.target_transform(off_t)
            frm_t = self.target_transform(frm_t)

        labels = {"on": on_t, "off": off_t, "frame": frm_t}

        if self.return_time_labels:
            if self.time_roll_rate is None:
                raise ValueError("Set time_roll_rate when return_time_labels=True.")
            roll_t = torch.tensor(
                pm.get_piano_roll(fs=self.time_roll_rate),
                dtype=torch.float32
            )  # [128, T_time]
            labels["time_roll"] = roll_t

        beats_window = torch.tensor(
            beats[start_beat_idx:start_beat_idx + num_beats + 1],
            dtype=torch.float32
        )
        meta = {
            "sr": sr,
            "hop_length": self.hop_length,
            "beats": beats_window,
            "subs_per_beat": self.S,
            "start_beat_idx": start_beat_idx,
        }

        mel = mel_for_model.to(torch.float32).unsqueeze(0)  # [1, n_mels, L]
        return mel, labels, meta