from typing import Dict, List, Tuple

from mamba_ssm import Mamba2
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_refine(batch: List[Dict]) -> Dict:
    """
    Collate variable-length samples from RefineDataset into a padded batch.

    T (spectrogram frames) varies per sample because beat-window duration
    depends on tempo.  L (beat-label steps) is fixed = beats_per_window * S.
    Total sequence length T+L therefore varies; sequences are right-padded
    to the longest in the batch.  Padding positions receive type_id = -1.

    Args:
        batch: list of dicts from RefineDataset.__getitem__

    Returns dict with:
        "sequence"          [B, max_len, feature_dim]
        "type_ids"          [B, max_len]  long  (-1 = padding)
        "lengths"           [B]           long  (true T_i + L per sample)
        "normalized_labels" {k: [B, L, 128]}    (L is same for all samples)
        "original_labels"   {k: [B, max_T, 128]}(T_i varies; zero-padded)
    """
    sequences = [s["sequence"] for s in batch]   # list of [T_i+L, D]
    type_ids  = [s["type_ids"] for s in batch]   # list of [T_i+L]

    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)

    seq_padded  = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # [B, max_len, D]
    type_padded = pad_sequence(type_ids,  batch_first=True, padding_value=-1)   # [B, max_len]

    # normalized_labels: L is fixed across the dataset
    label_keys  = ("on", "off", "frame", "velocity")
    norm_labels = {
        k: torch.stack([s["normalized_labels"][k] for s in batch])  # [B, L, 128]
        for k in label_keys
    }

    # original_labels: T varies — zero-pad to max T in this batch
    max_T = max(s["original_labels"]["frame"].shape[0] for s in batch)
    orig_labels: Dict[str, torch.Tensor] = {}
    for k in label_keys:
        padded = torch.zeros(len(batch), max_T, 128)
        for i, s in enumerate(batch):
            t = s["original_labels"][k]
            padded[i, : t.shape[0]] = t
        orig_labels[k] = padded  # [B, max_T, 128]

    return {
        "sequence":          seq_padded,   # [B, max_len, feature_dim]
        "type_ids":          type_padded,  # [B, max_len]
        "lengths":           lengths,      # [B]
        "normalized_labels": norm_labels,  # {k: [B, L, 128]}
        "original_labels":   orig_labels,  # {k: [B, max_T, 128]}
    }


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------

class OutputHead(nn.Module):
    """
    Per-token prediction head operating on Mamba output tokens.

    Key naming matches cache_spec.py / RefineDataset label convention:
        "on"       onset probability        (sigmoid)
        "off"      offset probability       (sigmoid)
        "frame"    active-note indicator    (sigmoid)
        "velocity" MIDI velocity 0-127      (raw linear — scale/clip in loss)
    """

    def __init__(self, dim: int, n_pitches: int = 128) -> None:
        super().__init__()
        self.on_head       = nn.Linear(dim, n_pitches)
        self.off_head      = nn.Linear(dim, n_pitches)
        self.frame_head    = nn.Linear(dim, n_pitches)
        self.velocity_head = nn.Linear(dim, n_pitches)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [..., dim]  →  dict of [..., n_pitches]"""
        return {
            "on":       torch.sigmoid(self.on_head(x)),
            "off":      torch.sigmoid(self.off_head(x)),
            "frame":    torch.sigmoid(self.frame_head(x)),
            "velocity": self.velocity_head(x),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FineAMT(nn.Module):
    """
    Mamba2-based AMT model that processes RefineDataset sequences.

    Input sequence layout (per sample):
        positions 0 … T-1   type_id=0  original-time spectrogram tokens
        positions T … T+L-1 type_id=1  beat-normalised label tokens
    T varies per sample (tempo-dependent); L is fixed.

    Two independent output heads decode the same Mamba output:
        coarse_head  →  supervised on original_labels  (type_id == 0)
        fine_head    →  supervised on normalized_labels (type_id == 1)

    Both heads run over all positions; outputs at irrelevant / padding
    positions are zeroed so the loss function can sum without masking.
    """

    def __init__(
        self,
        blocks: int,
        dim: int,
        feature_dim: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_pitches: int = 128,
    ) -> None:
        super().__init__()

        # Project dataset feature_dim → model dim (identity when equal)
        self.input_proj: nn.Module = (
            nn.Linear(feature_dim, dim, bias=False)
            if feature_dim != dim else nn.Identity()
        )

        # Learned type embeddings: index 0 = spec token, index 1 = beat token
        # Padding (type_id = -1) is clamped to 0 then zeroed by padding_mask.
        self.type_embedding = nn.Embedding(2, dim)

        self.mamba_blocks = nn.ModuleList([
            Mamba2(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(blocks)
        ])

        self.norm = nn.LayerNorm(dim)

        self.coarse_head = OutputHead(dim, n_pitches)  # original time grid
        self.fine_head   = OutputHead(dim, n_pitches)  # beat-synchronous grid

    def forward(
        self,
        batch: Dict,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            batch: dict from collate_refine
                "sequence"  [B, max_len, feature_dim]
                "type_ids"  [B, max_len]  (-1 for padding positions)

        Returns:
            coarse_out  dict {k: [B, max_len, n_pitches]}
                Non-zero only at positions where type_id == 0 (spec tokens).
            fine_out    dict {k: [B, max_len, n_pitches]}
                Non-zero only at positions where type_id == 1 (beat tokens).

        Loss computation:
            coarse_loss uses (type_ids == 0) mask against original_labels
            fine_loss   uses (type_ids == 1) mask against normalized_labels
        """
        x        = batch["sequence"]   # [B, max_len, feature_dim]
        type_ids = batch["type_ids"]   # [B, max_len]

        # Input projection + type embedding
        x = self.input_proj(x)                                     # [B, max_len, dim]
        x = x + self.type_embedding(type_ids.clamp(min=0))        # [B, max_len, dim]

        # Zero padding positions before Mamba so they don't pollute state
        padding_mask = (type_ids == -1).unsqueeze(-1)              # [B, max_len, 1]
        x = x.masked_fill(padding_mask, 0.0)

        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x)

        # Apply both heads to every position, then mask to their respective domains
        coarse_out = self.coarse_head(x)                           # {k: [B, max_len, n_pitches]}
        fine_out   = self.fine_head(x)

        coarse_mask = (type_ids == 0).unsqueeze(-1)                # [B, max_len, 1]
        fine_mask   = (type_ids == 1).unsqueeze(-1)

        coarse_out = {k: v * coarse_mask for k, v in coarse_out.items()}
        fine_out   = {k: v * fine_mask   for k, v in fine_out.items()}

        return coarse_out, fine_out