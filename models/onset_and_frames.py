import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from config import (SUBDIVISIONS_PER_BEAT, BEATS_PER_CLIP, BATCH_SIZE, N_MELS)

#d_model = dimensions of each embedding
class PatchEmbed2D(nn.Module):
    def __init__(self, n_mels:int=229, d_model:int=256, kernel_t:int=31, same:bool=True):
        super().__init__()
        self.n_mels = n_mels
        if same:
            # Preserves length for any odd kernel_t. For even, we add one extra pad on the right in forward().
            pad_t = (kernel_t - 1) // 2
        else:
            pad_t = 0

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(n_mels, kernel_t),   # full-band token
            stride=(n_mels, 1),               # F_out = 1
            padding=(0, pad_t),               # time padding only
            bias=True,
        )
        self.kernel_t = kernel_t
        self.same = same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mels, L]
        if self.same and self.kernel_t % 2 == 0:
            # asymmetric pad one step on the right to keep T_out == T_in
            x = F.pad(x, (0, 1, 0, 0))  # pad last dim (time): (left=0, right=1)
        x = self.proj(x)                # [B, d_model, 1, L]
        x = x.squeeze(2).transpose(1, 2)# [B, L, d_model]  (now L==input L)
        return x

    
class EncoderLayer(nn.Module):
    def __init__(self: "EncoderLayer", embed_dim: int, num_heads: int, dropout: float=0.1, mlp_ratio:float=2.0) -> None:
        super().__init__() # type: ignore[call-arg]
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        
        hidden = int(mlp_ratio * embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, 
                x, 
                key_padding_mask:Optional[Tensor]=None, 
                attn_mask:Optional[Tensor]=None
                ) -> Tensor:
        # x: [B, T, D]
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.drop1(attn_out)         # residual
        x = self.norm1(x)

        # Feed-forward
        mlp_out = self.mlp(x)
        x = x + self.drop2(mlp_out)          # residual
        x = self.norm2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, D]

    def forward(self, x: Tensor) -> Tensor:  # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0)
    
class DepthwiseTimeSmoother(nn.Module):
    """Per-pitch temporal smoothing used to spread onset/offset context a few steps."""
    def __init__(self, pitches: int = 128, k: int = 5):
        super().__init__()
        assert k % 2 == 1, "Use an odd kernel for 'same' padding"
        self.conv = nn.Conv1d(
            in_channels=pitches, out_channels=pitches,
            kernel_size=k, padding=k // 2, groups=pitches, bias=False
        )
        with torch.no_grad():
            w = torch.full((pitches, 1, k), 1.0 / k)
            self.conv.weight.copy_(w)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, P] -> conv over time dimension
        return self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, P]

class OnsetAndFrames(nn.Module) :
    def __init__(self:"OnsetAndFrames", 
                 d_model:int, 
                 n_heads:int, 
                 mlp_ratio:float=2.0, 
                 dropout:float=0.1, 
                 n_layers:int=1, 
                 n_notes:int=128,
                 smooth_k: int = 5,
                 detach_condition: bool = True) :
        super(OnsetAndFrames, self).__init__() # type: ignore[call-arg]
        kernel = SUBDIVISIONS_PER_BEAT
        self.patch = PatchEmbed2D(kernel_t=kernel)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # --- heads (logits) ---
        self.onset_head  = nn.Linear(d_model, n_notes)
        self.offset_head = nn.Linear(d_model, n_notes)
        self.frame_head  = nn.Linear(d_model, n_notes)
        
        # --- conditioning from on/off into frame ---
        self.on_smooth  = DepthwiseTimeSmoother(pitches=n_notes, k=smooth_k)
        self.off_smooth = DepthwiseTimeSmoother(pitches=n_notes, k=smooth_k)
        self.alpha_on   = nn.Parameter(torch.tensor(1.0))
        self.alpha_off  = nn.Parameter(torch.tensor(0.5))
        self.detach_condition = detach_condition
        
    def forward(self, mel_db: Tensor) -> Tensor:
        #The first layer takes in the spectrogram and splits it into patches using a CNN. 
        x = self.patch(mel_db)           # [B, N, D]
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)                 # [B, N, D]
        
        on_logits  = self.onset_head(x)     # [B, L, 128]
        off_logits = self.offset_head(x)    # [B, L, 128]
        frm_base   = self.frame_head(x)     # [B, L, 128]

        # Use onset/offset probs as features for frames
        on_prob  = torch.sigmoid(on_logits)
        off_prob = torch.sigmoid(off_logits)
        if self.detach_condition:
            on_prob  = on_prob.detach()
            off_prob = off_prob.detach()

        on_ctx  = self.on_smooth(on_prob)   # [B, L, 128]
        off_ctx = self.off_smooth(off_prob) # [B, L, 128]

        frame_logits = frm_base + self.alpha_on * on_ctx + self.alpha_off * off_ctx

        return {"on": on_logits, "off": off_logits, "frame": frame_logits}