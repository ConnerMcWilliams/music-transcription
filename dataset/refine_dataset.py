import pickle
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _load_pickle(path: str):
    """Load a pickle file from disk (cacheable helper)."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


class _PickleCache:
    """Thread-safe LRU cache for per-piece pickle files.

    Many windows share the same orig_spec_path, so caching avoids redundant
    disk reads.  *maxsize* controls memory usage — set to 0 to disable
    caching entirely.
    """

    def __init__(self, maxsize: int = 128) -> None:
        if maxsize > 0:
            self._get = lru_cache(maxsize=maxsize)(_load_pickle)
        else:
            self._get = _load_pickle  # type: ignore[assignment]

    def __call__(self, path: str):
        return self._get(path)


class RefineDataset(Dataset):
    """
    Dataset that pairs a non-normalized spectrogram window with variable-length
    beat-normalized MIDI labels into a single interleaved sequence for
    Mamba / Transformer AMT models.

    Each sample combines:
        - T spectrogram frames  (original time domain, fixed hop)   → type_id 0
        - x beat-MIDI steps     (beat-synchronous, from cache_spec) → type_id 1
    interleaved chronologically as a sequence of length T + x.

    Beat-MIDI tokens are already filtered to the spectrogram window by
    cache_spec.py, so no additional time filtering is needed.  Tokens are
    inserted at their correct time positions among the spectrogram frames.
    When a beat subdivision lands exactly on a spec frame edge, the spec
    frame appears first (stable sort).

    Metadata format
    ---------------
    ``metadata`` must be a list of dicts, one per window, with keys:

        "midi_path"       : str  — cache_spec.py output: midi_{idx}.pkl
                                   {"on","off","frame"} each [128, L]
        "ts_target_path"  : str  — cache_spec.py output: ts_target_{idx}.pkl
                                   [L] float32  beat-grid times in seconds
        "orig_spec_path"  : str  — per-piece spectrogram: {fname}.pkl
                                   [n_mels, T_full] or [1, n_mels, T_full] float32
        "start_frame"     : int  — first frame index (inclusive) into T_full
        "end_frame"       : int  — last frame index  (exclusive)  into T_full

    __getitem__ returns
    -------------------
        "sequence"     : Tensor [T + x, max_dim]
            Spectrogram and beat-MIDI tokens interleaved in chronological
            order.  x = number of beat subdivisions within the spec window.
        "type_ids"     : Tensor [T + x]  long
            0 for spectrogram frames, 1 for beat-MIDI steps.
        "midi_labels"  : dict {"on","off","frame"} each [x, 128]
            Beat-synchronized MIDI labels (transposed from [128, L] to [L, 128]).
    """

    # Keys to concatenate for the MIDI token feature vector, in order
    _LABEL_KEYS: Tuple[str, ...] = ("on", "off", "frame")

    def __init__(
        self,
        metadata: List[Dict],
        feature_dim: int = 512,
        n_mels: int = 128,
        label_pitch_dim: int = 128,
        dt: float = 0.02,
        cache_size: int = 512,
    ) -> None:
        super().__init__()
        self.metadata          = list(metadata)
        self.feature_dim       = feature_dim
        self.n_mels            = n_mels
        self.dt                = dt              # seconds per spec frame (hop_length / sample_rate)
        self.label_feature_dim = 3 * label_pitch_dim   # on + off + frame
        self._cache            = _PickleCache(maxsize=cache_size)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.metadata[index]

        # Load data
        spec_window = self._load_spec(sample)               # [T, n_mels]
        midi_labels = self._load_midi_labels(sample)         # dict {"on","off","frame"} each [x, 128]
        ts_target   = self._load_beat_times(sample)          # [x]

        # Build raw feature vectors (no projection — model handles that on GPU)
        spec_tokens = spec_window                             # [T, n_mels]
        label_tokens = torch.cat(
            [midi_labels[k] for k in self._LABEL_KEYS], dim=-1
        )  # [x, 3*P]

        T = spec_tokens.shape[0]
        x = label_tokens.shape[0]

        # Compute time positions in frame units for chronological interleaving
        start_frame = float(sample["start_frame"])
        spec_pos = torch.arange(T, dtype=torch.float32) + start_frame  # [T]
        beat_pos = ts_target / self.dt                                 # [x]

        # Pad spec and label tokens to the same width for concatenation.
        # The model's spec_proj / label_proj will handle the real projection.
        n_mels = self.n_mels
        label_dim = self.label_feature_dim
        max_dim = max(n_mels, label_dim)
        if n_mels < max_dim:
            spec_tokens = F.pad(spec_tokens, (0, max_dim - n_mels))    # [T, max_dim]
        if label_dim < max_dim:
            label_tokens = F.pad(label_tokens, (0, max_dim - label_dim))  # [x, max_dim]

        # Merge: spec tokens first in the concat so stable sort places
        # spec before beat when they land on the same frame edge.
        positions = torch.cat([spec_pos, beat_pos])                    # [T + x]
        order = torch.argsort(positions, stable=True)                  # [T + x]

        sequence = torch.cat([spec_tokens, label_tokens], dim=0)[order]  # [T + x, max_dim]
        type_ids = torch.cat([
            torch.zeros(T, dtype=torch.long),
            torch.ones(x,  dtype=torch.long),
        ])[order]  # [T + x]

        return {
            "sequence":    sequence,     # [T + x, max_dim]
            "type_ids":    type_ids,     # [T + x]
            "midi_labels": midi_labels,  # {"on","off","frame"} each [x, 128]
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_spec(
        self,
        sample: Dict,
    ) -> torch.Tensor:
        """
        Load and slice the original (non-beat-normalized) spectrogram.

        Returns:
            spec_window : [T, n_mels]  float32
        """
        s = int(sample["start_frame"])
        e = int(sample["end_frame"])

        spec_raw = self._cache(sample["orig_spec_path"])

        spec = self._to_tensor(spec_raw)
        if spec.dim() == 3:
            spec = spec.squeeze(0)    # [n_mels, T_full]
        return spec[:, s:e].T         # [T, n_mels]

    def _load_midi_labels(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """
        Load the beat-normalized MIDI label dict produced by cache_spec.py.

        Returns:
            {"on","off","frame"} each [L, 128] float32
                (transposed from the [128, L] storage format)
        """
        labels = self._cache(sample["midi_path"])

        return {
            k: self._to_tensor(v).T.contiguous()
            for k, v in labels.items()
            if k in self._LABEL_KEYS
        }

    def _load_beat_times(self, sample: Dict) -> torch.Tensor:
        """
        Load beat-grid subdivision times from ts_target pickle written by
        cache_spec.py.

        Returns:
            ts_target : [L] float32  beat-grid times in seconds
        """
        return self._to_tensor(self._cache(sample["ts_target_path"]))

    # ------------------------------------------------------------------
    # Tensor coercion utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(v) -> torch.Tensor:
        if isinstance(v, (tuple, list)):
            # conv_wav2fe saves (tensor, hop_length) tuples — unwrap.
            if len(v) > 0 and isinstance(v[0], (torch.Tensor, np.ndarray)):
                v = v[0]
            else:
                return torch.tensor(v, dtype=torch.float32)
        if isinstance(v, torch.Tensor):
            return v.float()
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float()
        return torch.tensor(v, dtype=torch.float32)