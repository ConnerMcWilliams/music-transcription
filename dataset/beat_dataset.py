# dataset_beats.py
import os
from typing import Optional, Callable, Tuple, Dict, Any

import torch
import torchaudio
from torch.utils.data import Dataset
from pretty_midi import PrettyMIDI

from dataset.transforms import linear_time_warp_mel, midi_notes_to_beat_labels
from utils.beat_utils import get_beats_and_downbeats, build_subdivision_times


class MaestroDatasetWithWindowingInBeats(Dataset):
    """
    Returns ONE beat-synchronous window per __getitem__:

      (mel_beat  [1, n_mels, L],  # L = beats_per_window * subdivisions
       labels    { 'on': [128, L], 'off': [128, L], 'frame': [128, L] },
       meta      { 'sr', 'hop_length', 'beats', 'subs_per_beat', 'start_beat_idx' })

    Notes:
      - `mel_tx` must return (mel_time: [n_mels, T], hop_length: int) when called as mel_tx(waveform, sr).
      - Tempo normalization (time→beat warping) happens here to preserve alignment with labels.
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
                 hop_length: int = None,
                 sr_target: Optional[int] = None,
                 waveform_tx: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 mel_aug_prewarp: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 mel_aug_postwarp: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 return_time_labels: bool = False,
                 time_roll_rate: Optional[float] = None   # fs for PrettyMIDI.get_piano_roll if return_time_labels
                 ):
        # --- plumbing / configuration ---
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
        self.dt = self.hop_length / self.sr_target  # seconds per mel frame

        self.return_time_labels = return_time_labels
        self.time_roll_rate = float(time_roll_rate) if time_roll_rate is not None else None

        # built index of (row_idx, start_beat_idx)
        self.index = []
        self._probe_and_build_index()

    # ---------- index building ----------
    def _probe_and_build_index(self) -> None:
        idx = []
        for i, row in self.metadata.iterrows():
            midi_path = os.path.join(self.root_dir, row["midi_filename"])
            try:
                pm = PrettyMIDI(midi_path)
            except Exception:
                continue

            beats, _ = get_beats_and_downbeats(pm)   # np.ndarray of beat times (sec)
            # need complete beat intervals; K = number of (beat_k → beat_{k+1}) segments
            K = max(0, len(beats) - 1)
            if K < self.beats_per_window:
                continue

            # start indices spaced by hop_beats, plus a final one to cover the tail exactly once
            starts = list(range(0, K - self.beats_per_window + 1, self.hop_beats))
            last_start = K - self.beats_per_window
            if not starts or starts[-1] != last_start:
                starts.append(last_start)

            for s in starts:
                idx.append((i, s))

        # de-duplicate (just in case) while preserving order
        self.index = list(dict.fromkeys(idx))

    def __len__(self) -> int:
        return len(self.index)

    # ---------- main fetch ----------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        row_idx, start_beat_idx = self.index[idx]
        row = self.metadata.iloc[row_idx]
        audio_path = os.path.join(self.root_dir, row["audio_filename"])
        midi_path  = os.path.join(self.root_dir, row["midi_filename"])

        # ---- load MIDI & define beat window ----
        pm = PrettyMIDI(midi_path)
        beats, _ = get_beats_and_downbeats(pm)
        num_beats = self.beats_per_window
        L = num_beats * self.S

        ts_target = build_subdivision_times(beats, self.S, start_beat_idx, num_beats)  # [L] in seconds
        if ts_target is None:
            # Degenerate; return empties with shapes intact
            empty_mel = torch.zeros(self.n_mels, L)
            empty = {'on': torch.zeros(128, L), 'off': torch.zeros(128, L), 'frame': torch.zeros(128, L)}
            meta = {'sr': None, 'hop_length': None, 'beats': None, 'subs_per_beat': self.S,
                    'start_beat_idx': start_beat_idx}
            return empty_mel.unsqueeze(0), empty, meta

        # ---- load audio (mono), optional resample ----
        waveform, sr = torchaudio.load(audio_path)  # [C, N]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.sr_target is not None and self.sr_target != sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr_target)
            sr = self.sr_target

        # ---- optional waveform-domain augmentation (no timing changes) ----
        if self.waveform_tx is not None:
            waveform = self.waveform_tx(waveform)  # [1, N]

        # ---- mel in time, then (optional) mel-domain pre-warp aug ----
        mel_time = self.mel_tx(waveform)  # [n_mels, T], int
        if mel_time.shape[0] != self.n_mels:
            raise ValueError(f"mel_tx returned {mel_time.shape[0]} mels, expected {self.n_mels}")
        if self.mel_aug_prewarp is not None:
            mel_time = self.mel_aug_prewarp(mel_time)     # [n_mels, T]

        # ---- tempo normalization: time → beat grid ----
        mel_beat = linear_time_warp_mel(mel_time, ts_target, self.dt)  # [n_mels, L]
        if mel_beat.shape != (self.n_mels, L):
            raise RuntimeError(f"mel_beat has shape {tuple(mel_beat.shape)}, expected ({self.n_mels}, {L})")

        # ---- optional beat-domain augmentation ----
        if self.mel_aug_postwarp is not None:
            mel_beat = self.mel_aug_postwarp(mel_beat)    # [n_mels, L]

        # ---- labels on the same beat grid ----
        on_b, off_b, frm_b = midi_notes_to_beat_labels(pm, beats, self.S, start_beat_idx, num_beats)  # [128, L]
        if self.target_transform is not None:
            on_b  = self.target_transform(on_b)
            off_b = self.target_transform(off_b)
            frm_b = self.target_transform(frm_b)
        labels = {'on': on_b, 'off': off_b, 'frame': frm_b}

        # ---- (optional) also return time-space roll for auxiliary losses ----
        if self.return_time_labels:
            if self.time_roll_rate is None:
                raise ValueError("Set time_roll_rate when return_time_labels=True.")
            roll_t = torch.tensor(pm.get_piano_roll(fs=self.time_roll_rate), dtype=torch.float32)  # [128, T_time]
            labels['time_roll'] = roll_t

        # ---- meta for possible unwarping back to seconds later ----
        beats_window = torch.tensor(beats[start_beat_idx:start_beat_idx+num_beats+1], dtype=torch.float32)
        meta = {
            'sr': sr,
            'hop_length': self.hop_length,
            'beats': beats_window,       # beat boundaries (seconds) for this window
            'subs_per_beat': self.S,
            'start_beat_idx': start_beat_idx
        }

        # Return mel as [1, n_mels, L] for 2D models (in_channels=1)
        mel_beat = mel_beat.unsqueeze(0)  # -> [1, n_mels, L]
        return mel_beat, labels, meta
