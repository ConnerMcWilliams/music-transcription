from mamba_ssm import Mamba2
import torch.nn as nn
import torch

class MambaEndToEnd() :
    def __init__(self, d_model, d_state, d_conv, expand, blocks) :
        self.mamba_blocks = [
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) 
            for _ in range(blocks)
        ]
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x) :
        # x: [BATCH, N_SLICES, FREQ, TIME]
        for mamba_block in self.mamba_blocks :
            x = mamba_block(x)
        x = self.norm(x)
            
        return x