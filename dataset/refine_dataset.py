import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
        "normalized_labels" : dict {"on","off","frame","velocity"} each [x, 128]
            Beat-synchronized labels for the beat-domain loss (filtered to
            subdivisions within the spectrogram window).
        "original_labels"   : dict {"on","off","frame","velocity"} each [T, 128]
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
    ) -> None:
        super().__init__()
        self.metadata          = list(metadata)
        self.feature_dim       = feature_dim
        self.dt                = dt              # seconds per spec frame (hop_length / sample_rate)
        self.label_feature_dim = 4 * label_pitch_dim   # on + off + frame + velocity

        # Linear projection: n_mels → feature_dim  (None when dims already match)
        self.spec_proj: Optional[nn.Linear] = (
            nn.Linear(n_mels, feature_dim, bias=False)
            if n_mels != feature_dim else None
        )

        # Linear projection: 4*label_pitch_dim → feature_dim
        self.label_proj: Optional[nn.Linear] = (
            nn.Linear(self.label_feature_dim, feature_dim, bias=False)
            if self.label_feature_dim != feature_dim else None
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.metadata[index]

        # Load data
        spec_window, orig_labels = self._load_original(sample)   # [T, n_mels], dict [T, 128]
        norm_labels              = self._load_norm_labels(sample) # dict [L, 128]
        ts_target                = self._load_beat_times(sample)  # [L]

        # Build unified sequence tokens
        spec_tokens  = self._build_spec_tokens(spec_window)    # [T, feature_dim]
        label_tokens = self._build_label_tokens(norm_labels)   # [L, feature_dim]

        T = spec_tokens.shape[0]

        # Filter beat tokens to those within the spec window's time range
        start_frame = float(sample["start_frame"])
        end_frame   = float(sample["end_frame"])
        t_start = start_frame * self.dt
        t_end   = end_frame * self.dt

        in_range = (ts_target >= t_start) & (ts_target < t_end)  # [L]
        ts_target    = ts_target[in_range]                        # [x]
        label_tokens = label_tokens[in_range]                     # [x, feature_dim]
        norm_labels  = {k: v[in_range] for k, v in norm_labels.items()}  # [x, 128]

        x = label_tokens.shape[0]

        # Compute time positions in frame units for chronological interleaving
        spec_pos = torch.arange(T, dtype=torch.float32) + start_frame  # [T]
        beat_pos = ts_target / self.dt                                 # [x]

        # Merge: spec tokens first in the concat so stable sort places
        # spec before beat when they land on the same frame edge.
        positions = torch.cat([spec_pos, beat_pos])                    # [T + x]
        order = torch.argsort(positions, stable=True)                  # [T + x]

        sequence = torch.cat([spec_tokens, label_tokens], dim=0)[order]  # [T + x, feature_dim]
        type_ids = torch.cat([
            torch.zeros(T, dtype=torch.long),
            torch.ones(x,  dtype=torch.long),
        ])[order]  # [T + x]

        return {
            "sequence":          sequence,     # [T + x, feature_dim]
            "type_ids":          type_ids,     # [T + x]
            "normalized_labels": norm_labels,  # {"on","off","frame","velocity"} [x, 128]
            "original_labels":   orig_labels,  # {"on","off","frame","velocity"} [T, 128]
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
        with open(sample["orig_spec_path"], "rb") as fh:
            spec_raw = pickle.load(fh)

        spec = self._to_tensor(spec_raw)
        if spec.dim() == 3:
            spec = spec.squeeze(0)    # [n_mels, T_full]
        spec_window = spec[:, s:e].T  # [T, n_mels]

        # --- Labels ---
        with open(sample["orig_labels_path"], "rb") as fh:
            raw_labels = pickle.load(fh)

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
        with open(sample["norm_labels_path"], "rb") as fh:
            labels = pickle.load(fh)

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
        with open(sample["extra_path"], "rb") as fh:
            extra = pickle.load(fh)

        return self._to_tensor(extra["times"])

    # ------------------------------------------------------------------
    # Sequence construction helpers
    # ------------------------------------------------------------------

    def _build_spec_tokens(self, spec_window: torch.Tensor) -> torch.Tensor:
        """
        spec_window : [T, n_mels]
        Returns     : [T, feature_dim]
        """
        if self.spec_proj is None:
            return spec_window
        with torch.no_grad():
            return self.spec_proj(spec_window)

    def _build_label_tokens(
        self,
        norm_labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Concatenate on / off / frame / velocity along the feature axis,
        then project to feature_dim.

        Returns : [L, feature_dim]
        """
        feat = torch.cat(
            [norm_labels[k] for k in self._LABEL_KEYS], dim=-1
        )  # [L, 4 * label_pitch_dim]

        if self.label_proj is None:
            return feat
        with torch.no_grad():
            return self.label_proj(feat)

    # ------------------------------------------------------------------
    # Tensor coercion utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(v) -> torch.Tensor:
        if isinstance(v, (tuple, list)):
            v = v[0]
        if isinstance(v, torch.Tensor):
            return v.float()
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float()
        return torch.tensor(v, dtype=torch.float32)