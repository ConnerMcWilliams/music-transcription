import torch
import torch.nn as nn
import torch.nn.functional as f

import torch.nn as nn

#d_model = dimensions of each embedding
class PatchEmbed2D(nn.Module):
    def __init__(self, n_mels=256, d_model=256, patch_f=16, patch_t=10, stride_f=None, stride_t=None):
        super().__init__()
        stride_f = stride_f or patch_f
        stride_t = stride_t or patch_t
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(patch_f, patch_t),
            stride=(stride_f, stride_t)
        )
    def forward(self, x):
        # x: [B, 1, n_mels, frames]
        x = self.proj(x)               # [B, d_model, F', T']
        x = x.flatten(2).transpose(1,2) # [B, N_tokens = F'*T', d_model]
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, mlp_ratio=2.0):
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
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.drop1(attn_out)         # residual
        x = self.norm1(x)

        # Feed-forward
        mlp_out = self.mlp(x)
        x = x + self.drop2(mlp_out)          # residual
        x = self.norm2(x)
        return x


class BasicTransformerAMT(nn.Module) :
    def __init__(self, frames, patch_t, patch_f, d_model, n_heads, mlp_ratio=2.0, dropout=0.1, n_layers=1, n_notes=128) :
        super(BasicTransformerAMT, self).__init__()
        self.embedding_layer = PatchEmbed2D()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, mlp_ratio, dropout)
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