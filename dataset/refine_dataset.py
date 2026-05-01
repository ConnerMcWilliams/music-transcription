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

        # Index arrays are tiny — load eagerly so __len__ works and worker
        # processes don't need to re-read them.
        idx = np.load(os.path.join(split_dir, "index.npz"))
        self.spec_start = idx["spec_start"]
        self.spec_end   = idx["spec_end"]
        self.midi_start = idx["midi_start"]
        self.midi_end   = idx["midi_end"]

        self.max_dim = max(n_mels, 3 * label_pitch_dim)

        # Validate the spec column count once in the parent without retaining
        # the mmap handle: handles are reopened lazily per worker (see
        # ``_ensure_open``) to avoid segfaults from sharing numpy mmaps
        # across fork() boundaries.
        _spec_probe = np.load(os.path.join(split_dir, "spec.npy"), mmap_mode="r")
        spec_cols = _spec_probe.shape[1]
        del _spec_probe
        if spec_cols != n_mels:
            raise ValueError(
                f"spec.npy has {spec_cols} mel bins but RefineDataset was "
                f"constructed with n_mels={n_mels}"
            )

        self.spec = self.midi_on = self.midi_off = self.midi_frame = self.ts_target = None

    def _ensure_open(self) -> None:
        """Open numpy mmaps lazily so each DataLoader worker gets its own
        file handles — sharing mmaps across fork() can segfault workers."""
        if self.spec is not None:
            return
        d = self.split_dir
        self.spec       = np.load(os.path.join(d, "spec.npy"),       mmap_mode="r")
        self.midi_on    = np.load(os.path.join(d, "midi_on.npy"),    mmap_mode="r")
        self.midi_off   = np.load(os.path.join(d, "midi_off.npy"),   mmap_mode="r")
        self.midi_frame = np.load(os.path.join(d, "midi_frame.npy"), mmap_mode="r")
        self.ts_target  = np.load(os.path.join(d, "ts_target.npy"),  mmap_mode="r")

    def __len__(self) -> int:
        return int(self.spec_start.shape[0])

    def __getitem__(self, index: int) -> Dict[str, object]:
        self._ensure_open()
        s,  e  = int(self.spec_start[index]), int(self.spec_end[index])
        ms, me = int(self.midi_start[index]), int(self.midi_end[index])

        T = e - s
        x = me - ms

        midi_labels = {
            "on":    torch.from_numpy(self.midi_on   [ms:me].copy()),     # [x, 128]
            "off":   torch.from_numpy(self.midi_off  [ms:me].copy()),
            "frame": torch.from_numpy(self.midi_frame[ms:me].copy()),
        }
        ts = torch.from_numpy(self.ts_target[ms:me].copy())                # [x]

        max_dim   = self.max_dim
        total_len = T + x

        spec_pos = torch.arange(T, dtype=torch.float32)
        beat_pos = ts / self.dt                                            # window-relative
        order    = torch.argsort(torch.cat([spec_pos, beat_pos]), stable=True)  # [T + x]
        type_ids = (order >= T).long()

        # Allocate the final sequence directly. Label rows stay zero — they
        # will be filled in-place at training time with perturbed labels.
        sequence = torch.zeros(total_len, max_dim)
        spec_dst = (type_ids == 0).nonzero(as_tuple=True)[0]
        sequence[spec_dst, :self.n_mels] = torch.from_numpy(self.spec[s:e].copy())

        return {
            "sequence":    sequence,
            "type_ids":    type_ids,
            "midi_labels": midi_labels,
        }
