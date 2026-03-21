from mamba_ssm import Mamba2
import torch
import torch.nn as nn


class OutputHead(nn.Module):
    def __init__(self, input_size: int, dim: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, dim)
        self.velocity_linear = nn.Linear(dim, output_size)
        self.onset_linear = nn.Linear(dim, output_size)
        self.offset_linear = nn.Linear(dim, output_size)
        self.frames_linear = nn.Linear(dim, output_size)

    def forward(self, x: torch.Tensor):
        """
        x: [..., input_size]
        returns:
            velocity: [..., output_size]
            frames:   [..., output_size]
            onset:    [..., output_size]
            offset:   [..., output_size]
        """
        x = self.linear(x)

        velocity = self.velocity_linear(x)
        frames = self.frames_linear(x)
        onset = self.onset_linear(x)
        offset = self.offset_linear(x)

        frames = torch.sigmoid(frames)
        onset = torch.sigmoid(onset)
        offset = torch.sigmoid(offset)

        return {
            "velocity": velocity,
            "frames": frames,
            "onset": onset,
            "offset": offset,
        }


class FineAMT(nn.Module):
    def __init__(
        self,
        blocks: int,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_pitches: int = 128,
        n_mels: int = 128,
    ):
        super().__init__()

        self.mamba_blocks = nn.ModuleList([
            Mamba2(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(blocks)
        ])

        self.coarse_head = OutputHead(dim, dim, n_pitches)
        self.first_output = OutputHead(dim, dim, n_pitches)

    def forward(self, x: torch.Tensor, type_IDs: torch.Tensor):
        """
        x:        [B, L, dim]
        type_IDs: [B, L] with 0 = coarse, 1 = fine
        """
        for block in self.mamba_blocks:
            x = block(x)

        coarse_mask = (type_IDs == 0)   # [B, L]
        fine_mask = (type_IDs == 1)     # [B, L]

        # Boolean indexing collapses dimensions, so result becomes [N_coarse, dim]
        coarse_tokens = x[coarse_mask]
        fine_tokens = x[fine_mask]

        coarse_output = self.coarse_head(coarse_tokens)
        first_output = self.first_output(fine_tokens)

        return coarse_output, first_output