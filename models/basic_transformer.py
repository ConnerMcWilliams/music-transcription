import torch
import torch.nn as nn
import torch.nn.functional as f

import torch.nn as nn

#d_model = dimensions of each embedding
class PatchEmbed2D(nn.Module):
    def __init__(self, d_model=256, kernel_t=32):
        super().__init__()
        pad_t = (kernel_t - 2) // 2        # ensures T_out = 500 when T_in = 501
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(229, kernel_t),    # span all 229 mel bins, big time kernel
            stride=(229, 1),                # F_out = 1, T_out = 500
            padding=(0, pad_t),
            bias=True,
        )

    def forward(self, x):                   # x: [B, 1, 229, 501]
        x = self.proj(x)                    # [B, d_model, 1, 500]
        x = x.squeeze(2).transpose(1, 2)    # [B, 500, d_model]
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, mlp_ratio=2.0):
        super().__init__()
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
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
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
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, D]

    def forward(self, x):               # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class BasicTransformerAMT(nn.Module) :
    def __init__(self, frames, patch_t, patch_f, d_model, n_heads, mlp_ratio=2.0, dropout=0.1, n_layers=1, n_notes=128) :
        super(BasicTransformerAMT, self).__init__()
        self.patch = PatchEmbed2D()
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_notes)
        
    def forward(self, mel_db):
        x = self.patch(mel_db)           # [B, N, D]
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)                 # [B, N, D]
        logits = self.head(x)            # [B, N, 128]
        # If patch_t>1, N != T; you'll need to map tokens back to frames, e.g. repeat/upsample or design patch_t=1.
        return logits