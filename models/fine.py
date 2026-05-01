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
# Mixture of Experts
# ---------------------------------------------------------------------------

class SparseMoE(nn.Module):
    """Top-k sparse Mixture of Experts feedforward layer."""

    def __init__(self, dim: int, n_experts: int = 8, top_k: int = 2, ffn_expand: int = 4) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * ffn_expand),
                nn.GELU(),
                nn.Linear(dim * ffn_expand, dim),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        flat = x.view(-1, D)                                      # [N, D]
        scores = self.gate(flat).softmax(-1)                      # [N, E]
        weights, indices = scores.topk(self.top_k, dim=-1)        # [N, k]
        weights = weights / weights.sum(-1, keepdim=True)         # normalise

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            # which tokens route to expert i (in any top-k slot)
            token_mask = (indices == i).any(-1)                   # [N]
            if not token_mask.any():
                continue
            # sum the weights for all k slots that point to expert i
            slot_mask = (indices[token_mask] == i).float()        # [n_sel, k]
            w = (weights[token_mask] * slot_mask).sum(-1, keepdim=True)  # [n_sel, 1]
            out[token_mask] += w * expert(flat[token_mask])
        return out.view(B, L, D)


# ---------------------------------------------------------------------------
# Jamba blocks
# ---------------------------------------------------------------------------

class JambaSSMBlock(nn.Module):
    """Pre-norm residual block: Mamba2 SSM + Sparse MoE FFN."""

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int,
                 n_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.ssm   = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.RMSNorm(dim)
        self.moe   = SparseMoE(dim, n_experts=n_experts, top_k=top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class JambaAttentionBlock(nn.Module):
    """Pre-norm residual block: Multi-head self-attention + Sparse MoE FFN."""

    def __init__(self, dim: int, n_heads: int = 8,
                 n_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        self.moe   = SparseMoE(dim, n_experts=n_experts, top_k=top_k)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.moe(self.norm2(x))
        return x


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
        dim: int,
        n_mels: int = 128,
        label_pitch_dim: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_pitches: int = 128,
        max_len: int = 4096,
        n_heads: int = 8,
        n_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()

        self.embedder = MixedTokenEmbedder(
            d1=n_mels,
            d2=3 * label_pitch_dim,   # on + off + frame
            d_model=dim,
            max_len=max_len,
        )
        
        

        ssm_block = lambda: JambaSSMBlock(dim, d_state, d_conv, expand, n_experts, top_k)
        self.jamba_block = nn.ModuleList(
            [ssm_block() for _ in range(3)]
            + [JambaAttentionBlock(dim, n_heads, n_experts, top_k)]
            + [ssm_block() for _ in range(3)]
        )

        self.norm = nn.LayerNorm(dim)
        self.fine_head       = OutputHead(dim, n_pitches)
        self.correction_head = OutputHead(dim, n_pitches)

    def forward(
        self,
        batch: Dict,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            batch: dict from collate_refine
                "sequence"  [B, max_len, max_dim]
                "type_ids"  [B, max_len]  (-1 for padding positions)

        Returns:
            {
              "fine":       {k: [B, max_len, n_pitches]} — main predictions
              "correction": {k: [B, max_len, n_pitches]} — denoised labels
            }
            Both are non-zero only at positions where type_id == 1 (beat tokens).
        """
        x        = batch["sequence"]   # [B, max_len, max_dim]
        type_ids = batch["type_ids"]   # [B, max_len]

        # Project both token types to d_model and add type/pos embeddings
        x = self.embedder(x, type_ids)  # [B, max_len, dim]

        # Zero padding positions before the blocks so they don't pollute state
        padding_mask = (type_ids == -1).unsqueeze(-1)  # [B, max_len, 1]
        x = x.masked_fill(padding_mask, 0.0)

        padding_bool = (type_ids == -1)  # [B, max_len]  True = pad
        for block in self.jamba_block:
            if isinstance(block, JambaAttentionBlock):
                x = block(x, key_padding_mask=padding_bool)
            else:
                x = block(x)
        x = self.norm(x)

        beat_mask = (type_ids == 1).unsqueeze(-1)
        fine_out       = {k: v * beat_mask for k, v in self.fine_head(x).items()}
        correction_out = {k: v * beat_mask for k, v in self.correction_head(x).items()}

        return {"fine": fine_out, "correction": correction_out}