"""
RefineDataset — consumes the monolithic per-split arrays written by
dataset/build_dataset.py. Each ``__getitem__`` is a handful of slices into
memory-mapped numpy buffers — no pickle loads, no LRU cache.

Expected layout for ``split_dir``:
    spec.npy        float32 [total_T, n_mels]
    midi_on.npy     float32 [total_x, 128]
    midi_off.npy    float32 [total_x, 128]
    midi_frame.npy  float32 [total_x, 128]
    ts_target.npy   float32 [total_x]      (window-relative seconds)
    index.npz       int64 arrays spec_start, spec_end, midi_start, midi_end
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class RefineDataset(Dataset):
    """
    Dataset that pairs a non-normalized spectrogram window with variable-length
    beat-normalized MIDI labels into a single interleaved sequence.

    Each sample combines:
        - T spectrogram frames                              → type_id 0
        - x beat-MIDI steps  (already filtered to window)   → type_id 1
    interleaved chronologically as a sequence of length T + x.

    When a beat subdivision lands exactly on a spec frame edge, the spec frame
    appears first (stable sort).

    __getitem__ returns
    -------------------
        "sequence"     : Tensor [T + x, max_dim]
        "type_ids"     : Tensor [T + x]            long
        "midi_labels"  : dict {"on","off","frame"} each [x, 128]
    """

    def __init__(
        self,
        split_dir: str,
        n_mels: int = 128,
        label_pitch_dim: int = 128,
        dt: float = 0.02,
    ) -> None:
        super().__init__()
        self.split_dir         = split_dir
        self.n_mels            = n_mels
        self.label_feature_dim = 3 * label_pitch_dim
        self.dt                = dt

        self.spec       = np.load(os.path.join(split_dir, "spec.npy"),       mmap_mode="r")
        self.midi_on    = np.load(os.path.join(split_dir, "midi_on.npy"),    mmap_mode="r")
        self.midi_off   = np.load(os.path.join(split_dir, "midi_off.npy"),   mmap_mode="r")
        self.midi_frame = np.load(os.path.join(split_dir, "midi_frame.npy"), mmap_mode="r")
        self.ts_target  = np.load(os.path.join(split_dir, "ts_target.npy"),  mmap_mode="r")

        idx = np.load(os.path.join(split_dir, "index.npz"))
        self.spec_start = idx["spec_start"]
        self.spec_end   = idx["spec_end"]
        self.midi_start = idx["midi_start"]
        self.midi_end   = idx["midi_end"]

        self.max_dim = max(n_mels, 3 * label_pitch_dim)

        if self.spec.shape[1] != n_mels:
            raise ValueError(
                f"spec.npy has {self.spec.shape[1]} mel bins but RefineDataset was "
                f"constructed with n_mels={n_mels}"
            )

    def __len__(self) -> int:
        return int(self.spec_start.shape[0])

    def __getitem__(self, index: int) -> Dict[str, object]:
        s,  e  = int(self.spec_start[index]), int(self.spec_end[index])
        ms, me = int(self.midi_start[index]), int(self.midi_end[index])

        T = e - s
        x = me - ms

        on  = torch.from_numpy(self.midi_on   [ms:me].copy())              # [x, 128]
        off = torch.from_numpy(self.midi_off  [ms:me].copy())
        frm = torch.from_numpy(self.midi_frame[ms:me].copy())
        ts  = torch.from_numpy(self.ts_target [ms:me].copy())              # [x]

        midi_labels = {"on": on, "off": off, "frame": frm}

        max_dim = self.max_dim

        spec_tokens = torch.zeros(T, max_dim)
        spec_tokens[:, :self.n_mels] = torch.from_numpy(self.spec[s:e].copy())

        label_tokens = torch.zeros(x, max_dim)
        label_tokens[:, :128]    = on
        label_tokens[:, 128:256] = off
        label_tokens[:, 256:384] = frm

        spec_pos  = torch.arange(T, dtype=torch.float32)
        beat_pos  = ts / self.dt                                           # window-relative
        order     = torch.argsort(torch.cat([spec_pos, beat_pos]), stable=True)  # [T + x]

        sequence  = torch.cat([spec_tokens, label_tokens], dim=0)[order]   # [T + x, max_dim]
        type_ids  = (order >= T).long()

        return {
            "sequence":    sequence,
            "type_ids":    type_ids,
            "midi_labels": midi_labels,
        }
