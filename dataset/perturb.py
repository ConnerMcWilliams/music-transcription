"""
Label-perturbation utilities for the RefineAMT correction task.

The model receives beat-grid MIDI tokens as input alongside the spectrogram
frames. With clean inputs, supervising the model on those same labels is
degenerate — it can copy. To force the model to *correct* errors, we randomly
corrupt the input labels and supervise a separate correction head on the
clean labels at the corrupted positions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def perturb_labels(
    midi_labels: Dict[str, torch.Tensor],
    midi_mask:   torch.Tensor,
    p_row:       float = 0.5,
    p_flip:      float = 0.03,
    generator:   Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Bit-flip a subset of beat-token labels.

    Args:
        midi_labels: dict {"on","off","frame"} each [B, max_x, 128] in {0,1}.
        midi_mask:   [B, max_x] bool — True at valid (non-padded) beat positions.
        p_row:       per-row probability that a row is *eligible* for flips.
        p_flip:      per-element flip probability inside an eligible row.
        generator:   optional torch.Generator for reproducibility.

    Returns:
        perturbed_labels: dict with same shape as ``midi_labels``.
        correction_mask:  [B, max_x] bool — True at rows where at least one
                          element actually flipped (these are the rows the
                          correction head should be supervised on).
    """
    keys = ("on", "off", "frame")
    on = midi_labels["on"]
    B, X, P = on.shape
    device = on.device

    # Stack once and operate on the [3, B, X, P] block — one random tensor,
    # one torch.where, one kernel launch instead of three.
    stacked = torch.stack([midi_labels[k] for k in keys], dim=0)            # [3, B, X, P]

    eligible = (torch.rand(B, X, device=device, generator=generator) < p_row) & midi_mask
    flip = (
        torch.rand(3, B, X, P, device=device, generator=generator) < p_flip
    ) & eligible.unsqueeze(-1)                                              # [3, B, X, P]

    out = torch.where(flip, 1.0 - stacked, stacked)
    perturbed = {k: out[i] for i, k in enumerate(keys)}

    correction_mask = flip.any(dim=-1).any(dim=0) & midi_mask
    return perturbed, correction_mask
