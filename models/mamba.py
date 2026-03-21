import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputHead(nn.Module) :
    def __init__(self, input_size, dim, output_size) :
        self.linear = nn.Linear(input_size, dim)
        self.velocity_linear = nn.Linear(dim, output_size)
        self.onset_linear = nn.Linear(dim, output_size)
        self.offset_linear = nn.Linear(dim, output_size)
        self.frames_linear = nn.Linear(dim, output_size)
        
    def forward(self, x) :
        x = self.linear(x)
        velocity = self.velocity_linear(x)
        frames = self.offset_linear(x)
        onset = self.onset_linear(x)
        offset = self.frames_linear(x)
        
        frames = F.sigmoid(frames)
        onset = F.sigmoid(onset)
        offset = F.sigmoid(offset)
        
        output = [velocity, frames, 
                         onset, offset]
        
        return output

class MambaBlock(nn.Module) :
    def __init__(self, dim, output_dim, kernel_size, delta) :
        
        def discretize_A(A, delta):
            return torch.exp(delta * A)
        
        def discretize_B(B, delta):
            return delta * B   # simple approximation
        
        self.embedGate = nn.Conv1d(dim, output_dim, kernel_size)
        self.embedSSM = nn.Conv1d(dim, output_dim, kernel_size)
        
        self.A1 = torch.zeros()
        self.A2 = None
        self.B1 = None
        self.B2 = None
        self.C1 = None
        self.C2 = None
        
        self.N = None
        self.delta = None
        self.discretize_A = discretize_A
        self.discretize_B = discretize_B
        
        self.relu = nn.ReLU()
        
        self.linear_layer = nn.Linear(dim, output_dim)
    
    def forward(self, x, type_ids):
        """
            input: X sequence length [t + L, n_mels]
            output: X sequence length [t + L, n_mels]
        """
        
        # x: [B, T, D_in]
        # type_ids: [B, T], 0=spec, 1=coarse

        B, T, _ = x.shape

        # projections
        gate_x = self.relu(self.embedGate(x))   # [B, T, N]
        z      = self.relu(self.embedSSM(x))    # [B, T, N]

        h = torch.zeros(B, self.N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            is_coarse_t = (type_ids[:, t] == 1)   # [B]

            A_t = torch.where(is_coarse_t[:, None], self.A_1[:, t], self.A_2[:, t])
            B_t = torch.where(is_coarse_t[:, None], self.B_1[:, t], self.B_2[:, t])
            C_t = torch.where(is_coarse_t[:, None], self.C_1[:, t], self.C_2[:, t])

            delta_t = self.delta[:, t]

            A_bar_t = self.discretize_A(A_t, delta_t)
            B_bar_t = self.discretize_B(B_t, delta_t)

            h = A_bar_t * h + B_bar_t * z[:, t]
            y_t = C_t * h
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, T, N]

        # apply gate only to spectrogram tokens
        spec_mask = (type_ids == 0)  # [B, T]
        gate = torch.sigmoid(gate_x)

        y = torch.where(spec_mask.unsqueeze(-1), y * gate, y)

        y = self.linear_out(y)
        return y
        

class MambaModel(nn.Module):
    
    def __init__(self, mamba_blocks, dim=256, n_pitches=128, n_mels=128):
        self.mamba_blocks = [MambaBlock() for _ in range(mamba_blocks)] # [t+L, n_mels]
        self.coarse_head = OutputHead(n_mels, dim, n_pitches) # [B, L, n_mels] -> 4 x [B, L, n_mels]
        self.first_output = OutputHead(n_mels, dim, n_pitches) 

    """
        input: 
            X sequence length [t + L, n_mels]
            type_IDs: [t + L]
        output: 
            Coarse Output:  [4, L, n_mels]
            Refined Output: [4, t, n_mels]
    """
    def forward(self, x, type_IDs) :
        for block in self.mamba_blocks :
            x = block(x, type_IDs) 
        
        coarse_tokens = x[type_IDs == 0]
        fine_tokens   = x[type_IDs == 1]
        
        coarse_output = self.coarse_head(coarse_tokens)
        first_output = self.first_output(fine_tokens)
        
        return coarse_output, first_output