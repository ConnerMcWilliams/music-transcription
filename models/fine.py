from typing import Dict, List, Tuple

from mamba_ssm import Mamba2
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class MixedTokenEmbedder(nn.Module):
    def __init__(self, d1, d2, d_model, max_len=4096):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.proj1 = nn.Sequential(
            nn.Linear(d1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(d2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, token_type_ids):
        """
        x: [B, L, max_dim]  zero-padded raw features
        token_type_ids: [B, L] with values 0 or 1 (use -1 for padding)
        """
        B, L = token_type_ids.shape
        pos_ids = torch.arange(L, device=token_type_ids.device).unsqueeze(0).expand(B, L)

        out = torch.zeros(B, L, self.type_embed.embedding_dim, device=x.device, dtype=x.dtype)

        mask1 = token_type_ids == 0
        mask2 = token_type_ids == 1

        if mask1.any():
            out[mask1] = self.proj1(x[mask1][:, :self.d1])

        if mask2.any():
            out[mask2] = self.proj2(x[mask2][:, :self.d2])

        out = out + self.type_embed(token_type_ids.clamp(min=0)) + self.pos_embed(pos_ids)
        out = self.norm(out)
        return out


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
    """

    def __init__(self, dim: int, n_pitches: int = 128) -> None:
        super().__init__()
        self.on_head    = nn.Linear(dim, n_pitches)
        self.off_head   = nn.Linear(dim, n_pitches)
        self.frame_head = nn.Linear(dim, n_pitches)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [..., dim]  →  dict of [..., n_pitches]"""
        return {
            "on":    torch.sigmoid(self.on_head(x)),
            "off":   torch.sigmoid(self.off_head(x)),
            "frame": torch.sigmoid(self.frame_head(x)),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FineAMT(nn.Module):
    """
    Mamba2-based AMT model that processes RefineDataset sequences.

    Input sequence layout (per sample):
        positions with type_id=0  →  original-time spectrogram tokens
        positions with type_id=1  →  beat-normalised MIDI tokens
    interleaved chronologically.

    MixedTokenEmbedder projects both token types to the same d_model,
    adding type and positional embeddings.

    Output head decodes beat-synchronous positions (type_id == 1) into
    on/off/frame predictions.
    """

    def __init__(
        self,
        blocks: int,
        dim: int,
        n_mels: int = 128,
        label_pitch_dim: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_pitches: int = 128,
        max_len: int = 4096,
    ) -> None:
        super().__init__()

        self.embedder = MixedTokenEmbedder(
            d1=n_mels,
            d2=3 * label_pitch_dim,   # on + off + frame
            d_model=dim,
            max_len=max_len,
        )

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
        self.fine_head = OutputHead(dim, n_pitches)

    def forward(
        self,
        batch: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict from collate_refine
                "sequence"  [B, max_len, max_dim]
                "type_ids"  [B, max_len]  (-1 for padding positions)

        Returns:
            fine_out    dict {k: [B, max_len, n_pitches]}
                Non-zero only at positions where type_id == 1 (beat tokens).
        """
        x        = batch["sequence"]   # [B, max_len, max_dim]
        type_ids = batch["type_ids"]   # [B, max_len]

        # Project both token types to d_model and add type/pos embeddings
        x = self.embedder(x, type_ids)  # [B, max_len, dim]

        # Zero padding positions before Mamba so they don't pollute state
        padding_mask = (type_ids == -1).unsqueeze(-1)  # [B, max_len, 1]
        x = x.masked_fill(padding_mask, 0.0)

        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x)

        # Decode only beat-synchronous positions
        fine_out = self.fine_head(x)
        fine_mask = (type_ids == 1).unsqueeze(-1)
        fine_out = {k: v * fine_mask for k, v in fine_out.items()}

        return fine_out