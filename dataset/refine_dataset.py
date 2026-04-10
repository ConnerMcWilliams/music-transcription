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

    Many windows share the same orig_spec_path / orig_labels_path, so caching
    avoids redundant disk reads.  *maxsize* controls memory usage — set to 0
    to disable caching entirely.
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
    Dataset that pairs a non-normalized spectrogram window with beat-normalized
    labels into a single sequence for Mamba / Transformer AMT models.

    Each sample combines:
        - T spectrogram frames  (original time domain, fixed hop)   → type_id 0
        - L beat-label steps    (beat-synchronous, from cache_spec) → type_id 1
    interleaved chronologically as a sequence of length T + x.

    Beat-label tokens are filtered to only those whose times fall within the
    spectrogram window [start_frame * dt, end_frame * dt), yielding x ≤ L
    tokens.  These are inserted at their correct time positions among the
    spectrogram frames.  For example, if beat 1 falls between spec frames 24
    and 25:  [..., t_24, b_1, t_25, ...].  When a beat subdivision lands
    exactly on a spec frame edge, the spec frame appears first (stable sort).

    Metadata format
    ---------------
    ``metadata`` must be a list of dicts, one per window, with keys:

        "norm_labels_path" : str  — cache_spec.py output: labels_{idx}.pkl
                                    {"on","off","frame","velocity"} each [L, 128]
        "extra_path"       : str  — cache_spec.py output: extra_{idx}.pkl
                                    {"times": [L] float32, "max_time": float|None}
        "orig_spec_path"   : str  — per-piece spectrogram: {fname}.pkl
                                    [n_mels, T_full] or [1, n_mels, T_full] float32
        "orig_labels_path" : str  — per-piece note2label.py pickle: {fname}.pkl
                                    {"mpe","onset","offset","velocity"} each [T_full, 128]
        "start_frame"      : int  — first frame index (inclusive) into T_full
        "end_frame"        : int  — last frame index  (exclusive)  into T_full

    __getitem__ returns
    -------------------
        "sequence"          : Tensor [T + x, feature_dim]
            Spectrogram and beat-label tokens interleaved in chronological
            order.  x = number of beat subdivisions within the spec window.
        "type_ids"          : Tensor [T + x]  long
            0 for spectrogram frames, 1 for beat-label steps.
        "normalized_labels" : dict {"on","off","frame","velocity"} each [x, P]
            Beat-synchronized labels for the beat-domain loss (filtered to
            subdivisions within the spectrogram window).
        "original_labels"   : dict {"on","off","frame","velocity"} each [T, P]
            Time-domain labels aligned with the spectrogram window.

    Projections
    -----------
    If n_mels != feature_dim, a Linear(n_mels, feature_dim) is applied to
    spectrogram tokens.  If 4*label_pitch_dim != feature_dim, a Linear
    (4*label_pitch_dim, feature_dim) is applied to label tokens.  Both are
    called inside torch.no_grad() — they are not differentiable through the
    Dataset.  To make them trainable, register dataset.spec_proj /
    dataset.label_proj as parameters in the model instead.
    """

    # Keys to concatenate for the label token feature vector, in order
    _LABEL_KEYS: Tuple[str, ...] = ("on", "off", "frame", "velocity")

    # Mapping from note2label.py raw keys → canonical output keys
    _RAW_KEY_MAP: Dict[str, str] = {
        "onset":    "on",
        "offset":   "off",
        "mpe":      "frame",
        "velocity": "velocity",
    }

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
        self.label_feature_dim = 4 * label_pitch_dim   # on + off + frame + velocity
        self._cache            = _PickleCache(maxsize=cache_size)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.metadata[index]

        # Load data
        spec_window, orig_labels = self._load_original(sample)   # [T, n_mels], dict [T, P]
        norm_labels              = self._load_norm_labels(sample) # dict [L, P]
        ts_target                = self._load_beat_times(sample)  # [L]

        # Build raw feature vectors (no projection — model handles that on GPU)
        spec_tokens = spec_window                                  # [T, n_mels]
        label_tokens = torch.cat(
            [norm_labels[k] for k in self._LABEL_KEYS], dim=-1
        )  # [L, 4*P]

        T = spec_tokens.shape[0]

        # Filter beat tokens to those within the spec window's time range
        start_frame = float(sample["start_frame"])
        end_frame   = float(sample["end_frame"])
        t_start = start_frame * self.dt
        t_end   = end_frame * self.dt

        in_range = (ts_target >= t_start) & (ts_target <= t_end)  # [L]
        ts_target    = ts_target[in_range]                        # [x]
        label_tokens = label_tokens[in_range]                     # [x, 4*P]
        norm_labels  = {k: v[in_range] for k, v in norm_labels.items()}  # [x, P]

        x = label_tokens.shape[0]

        # Compute time positions in frame units for chronological interleaving
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
            "sequence":          sequence,     # [T + x, max_dim]
            "type_ids":          type_ids,     # [T + x]
            "normalized_labels": norm_labels,  # {"on","off","frame","velocity"} [x, P]
            "original_labels":   orig_labels,  # {"on","off","frame","velocity"} [T, P]
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_original(
        self,
        sample: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load and slice the original (non-beat-normalized) spectrogram and labels.

        Returns:
            spec_window : [T, n_mels]  float32
            orig_labels : {"on","off","frame","velocity"} each [T, 128]
        """
        s = int(sample["start_frame"])
        e = int(sample["end_frame"])

        # --- Spectrogram ---
        spec_raw = self._cache(sample["orig_spec_path"])

        spec = self._to_tensor(spec_raw)
        if spec.dim() == 3:
            spec = spec.squeeze(0)    # [n_mels, T_full]
        spec_window = spec[:, s:e].T  # [T, n_mels]

        # --- Labels ---
        raw_labels = self._cache(sample["orig_labels_path"])

        orig_labels = {
            out_key: self._to_tensor(raw_labels[raw_key])[s:e]
            for raw_key, out_key in self._RAW_KEY_MAP.items()
        }

        return spec_window, orig_labels

    def _load_norm_labels(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """
        Load the beat-normalized label dict produced by cache_spec.py.

        Returns:
            {"on","off","frame","velocity"} each [L, 128] float32
        """
        labels = self._cache(sample["norm_labels_path"])

        return {
            k: self._to_tensor(v)
            for k, v in labels.items()
            if k in self._LABEL_KEYS
        }

    def _load_beat_times(self, sample: Dict) -> torch.Tensor:
        """
        Load beat-grid subdivision times from the extra pickle written by
        cache_spec.py.

        Returns:
            ts_target : [L] float32  beat-grid times in seconds
        """
        extra = self._cache(sample["extra_path"])

        return self._to_tensor(extra["times"])

    # ------------------------------------------------------------------
    # Tensor coercion utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(v) -> torch.Tensor:
        if isinstance(v, (tuple, list)):
            # conv_wav2fe saves (tensor, hop_length) tuples — unwrap.
            # note2label saves nested Python lists — convert directly.
            if len(v) > 0 and isinstance(v[0], (torch.Tensor, np.ndarray)):
                v = v[0]
            else:
                return torch.tensor(v, dtype=torch.float32)
        if isinstance(v, torch.Tensor):
            return v.float()
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float()
        return torch.tensor(v, dtype=torch.float32)