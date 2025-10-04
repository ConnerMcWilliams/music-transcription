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
        super().__init__() # type: ignore[call-arg]
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, D]

    def forward(self, x:Tensor)->Tensor:               # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class OnsetAndFrames(nn.Module) :
    def __init__(self:"OnsetAndFrames", 
                 d_model:int, 
                 n_heads:int, 
                 mlp_ratio:float=2.0, 
                 dropout:float=0.1, 
                 n_layers:int=1, 
                 n_notes:int=128) :
        super(OnsetAndFrames, self).__init__() # type: ignore[call-arg]
        kernel = SUBDIVISIONS_PER_BEAT
        self.patch = PatchEmbed2D(kernel_t=kernel)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.onsetHead = nn.Linear(d_model, n_notes)
        
    def forward(self, mel_db: Tensor) -> Tensor:
        #The first layer takes in the spectrogram and splits it into patches using a CNN. 
        x = self.patch(mel_db)           # [B, N, D]
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)                 # [B, N, D]
        logits = self.onsetHead(x)            # [B, N, 128]
        return logits