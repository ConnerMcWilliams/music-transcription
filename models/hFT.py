import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """
    Light 2D conv front-end.
    Preserves time resolution; may reduce frequency resolution a bit.
    Input:  [B, 1, T, F]
    Output: [B, C, T, F_out]
    """
    def __init__(self, in_ch=1, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),

            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            # Optional: downsample frequency only, not time
            nn.Conv2d(dim, dim, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AxisTransformerBlock(nn.Module):
    """
    Generic transformer block applied along one axis.
    Input: [N, L, D]
    Output: [N, L, D]
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x):
        # x: [N, L, D]
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class FreqTimeBlock(nn.Module):
    """
    One block of:
      1) frequency-axis attention
      2) time-axis attention

    Input / output: [B, C, T, F]
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, ff_mult=4):
        super().__init__()
        self.freq_block = AxisTransformerBlock(dim, num_heads, dropout, ff_mult)
        self.time_block = AxisTransformerBlock(dim, num_heads, dropout, ff_mult)

    def forward(self, x):
        B, C, T, F = x.shape

        # ----- Frequency-axis attention -----
        # For each time step, attend across frequency bins.
        # [B, C, T, F] -> [B, T, F, C] -> [(B*T), F, C]
        xf = x.permute(0, 2, 3, 1).reshape(B * T, F, C)
        xf = self.freq_block(xf)
        x = xf.reshape(B, T, F, C).permute(0, 3, 1, 2)  # back to [B, C, T, F]

        # ----- Time-axis attention -----
        # For each frequency bin, attend across time steps.
        # [B, C, T, F] -> [B, F, T, C] -> [(B*F), T, C]
        xt = x.permute(0, 3, 2, 1).reshape(B * F, T, C)
        xt = self.time_block(xt)
        x = xt.reshape(B, F, T, C).permute(0, 3, 2, 1)  # back to [B, C, T, F]

        return x


class AMTEncoder(nn.Module):
    """
    Example encoder:
      spectrogram -> conv stem -> stacked freq-time blocks -> projection
    """
    def __init__(
        self,
        in_ch=1,
        dim=256,
        depth=6,
        num_heads=8,
        dropout=0.1,
        ff_mult=4,
    ):
        super().__init__()
        self.stem = ConvStem(in_ch=in_ch, dim=dim)
        self.blocks = nn.ModuleList([
            FreqTimeBlock(dim, num_heads, dropout, ff_mult)
            for _ in range(depth)
        ])
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: [B, 1, T, F]
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x  # [B, C, T, F_out]
    
class PianoHeads(nn.Module):
    def __init__(self, dim, n_pitches=88):
        super().__init__()
        self.dim = dim
        self.n_pitches = n_pitches
        # Use LazyLinear for automatic input size detection
        self.proj = nn.LazyLinear(512)
        self.onset_head  = nn.Linear(512, n_pitches)
        self.frame_head  = nn.Linear(512, n_pitches)
        self.offset_head = nn.Linear(512, n_pitches)

    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, F = x.shape

        # flatten (C, F) at each time step
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        x = torch.nn.functional.gelu(self.proj(x))

        onset_logits  = self.onset_head(x)
        frame_logits  = self.frame_head(x)
        offset_logits = self.offset_head(x)

        return {
            "onset": onset_logits,
            "frame": frame_logits,
            "offset": offset_logits,
        }


class HFTModel(nn.Module):
    """
    Combined harmonic-frequency-time model for music transcription.
    Wraps AMTEncoder and PianoHeads with output key compatibility.
    
    Input: [B, 1, n_mels, time_steps] spectrogram (freq × time format)
    Output: {"on": onset_logits, "frame": frame_logits, "off": offset_logits}
    """
    def __init__(self, dim=256, depth=6, num_heads=8, dropout=0.1, ff_mult=4, n_pitches=128):
        super().__init__()
        self.encoder = AMTEncoder(
            in_ch=1,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            ff_mult=ff_mult,
        )
        self.heads = PianoHeads(dim=dim, n_pitches=n_pitches)

    def forward(self, x):
        # Input: [B, 1, n_mels, time_steps] (freq × time format from spectrogram)
        # Convert to [B, 1, time_steps, n_mels] (time × freq format expected by hFT)
        x = x.transpose(2, 3)
        
        x = self.encoder(x)  # [B, C, T, F]
        logits = self.heads(x)
        
        # Rename keys to match expected format: onset->on, offset->off
        return {
            "on": logits["onset"],
            "off": logits["offset"],
            "frame": logits["frame"],
        }