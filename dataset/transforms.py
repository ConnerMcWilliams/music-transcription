import torch
import torchaudio
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH,WIN_LENGTH, N_MELS,F_MIN,F_MAX,WINDOW_FN,POWER,LOG_OFFSET

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    power=POWER,                  
    center=True,
    pad_mode="reflect",
    window_fn=torch.hann_window,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    mel_scale="slaney",        
    norm="slaney"               
)

def log_mel(waveform):
    S = mel(waveform)          # [B, n_mels, T]
    return torch.log(S + LOG_OFFSET)  # or torch.log1p(S) if you prefer