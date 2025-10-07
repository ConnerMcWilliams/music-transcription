from dataset.transforms import log_mel
from utils.display_midi import display_spectrogram, display_midi_from_roll, display_spectrogram_with_beat
from experiment import run_experiment
from config import (DEVICE, MODEL_VARIANTS, NUM_EPOCHS,
                    CSV_PATH, MAESTRO_ROOT, BEATS_PER_CLIP, 
                    SUBDIVISIONS_PER_BEAT, HOP_LENGTH,
                    BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, DROP_LAST_TRAIN)
from train import load_or_compute_pos_weight
from dataset.beat_dataset import MaestroDatasetWithWindowingInBeats
import pandas as pd
from models.onset_and_frames import OnsetAndFrames
from data import make_loader
from models.loss import OnsetsAndFramesLoss
from schedulers import make_optimizer
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

LABEL_KEYS = ("on", "off", "frame", "vel")  # 'vel' only if you use it

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None, step_per_batch=True, device=DEVICE):
    model.train()
    total = 0.0
    lr_track = []

    for step, (x, labels) in enumerate(tqdm(loader, leave=False, desc="train")):
        # Move input
        x = x.to(device, dtype=torch.float32, non_blocking=True)

        # Move/prepare label dict WITHOUT permuting
        y = {}
        for k in LABEL_KEYS:
            if k in labels:
                v = labels[k]
                # Accept [L,128] or [B,L,128]; normalize to [B,L,128]
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                y[k] = v.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model(x)                           # dict: {'on','off','frame',(optional 'vel')}
        loss, _metrics = criterion(out, y)       # <-- pass the dict, not a tensor
        loss.backward()
        optimizer.step()

        if scheduler is not None and step_per_batch:
            scheduler.step()

        lr_track.append(optimizer.param_groups[0]["lr"])
        total += loss.item() * x.size(0)

        # Optional one-time sanity print
        if step == 0:
            print("x:", x.shape, x.dtype)
            for k in ("on","off","frame"):
                if k in y:
                    print(f"y[{k}]:", y[k].shape, y[k].dtype)
            for k in ("on","off","frame"):
                if k in out:
                    print(f"out[{k}]:", out[k].shape, out[k].dtype)

    avg_loss = total / max(1, len(loader.dataset))
    return avg_loss, lr_track

LABEL_KEYS = ("on", "off", "frame")

def collate_numeric(batch):
    """
    Each item: (mel: [1, n_mels, L] float, labels: dict[str, [L, 128] float], meta: dict)
    Returns:
      X: [B, 1, n_mels, L] float32
      Y: dict[str, [B, L, 128] float32]
    """
    Xs, Ys = [], []

    for item in batch:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError(f"Dataset item must be (mel, labels, ...). Got: {type(item)} / len={len(item) if isinstance(item,(tuple,list)) else 'n/a'}")

        mel, labels = item[0], item[1]

        # --- inputs
        if not torch.is_tensor(mel):
            mel = torch.as_tensor(mel)
        Xs.append(mel.to(torch.float32))

        # --- labels (only the three we train on)
        y = {}
        for k in LABEL_KEYS:
            v = labels[k]
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            # Accept [128, L] or [L, 128]; normalize to [L, 128]
            if v.dim() != 2:
                raise ValueError(f"labels['{k}'] must be 2D, got {v.shape}")
            if v.shape[0] == 128 and v.shape[1] != 128:
                v = v.transpose(0, 1)
            elif v.shape[1] != 128:
                raise ValueError(f"labels['{k}'] last dim must be 128 pitches, got {v.shape}")
            y[k] = v.to(torch.float32)
        Ys.append(y)

    X = torch.stack(Xs, 0)  # [B, 1, n_mels, L]
    Y = {k: torch.stack([y[k] for y in Ys], 0) for k in LABEL_KEYS}  # [B, L, 128]
    return X, Y


def main() :
    """
    Return the splits from 2017 and 2018
    """
    metadata = pd.read_csv(CSV_PATH)
    
    train_split = metadata[
        ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
        (metadata['split'] == 'train')
    ]
    
    val_split = metadata[
        ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
        (metadata['split'] == 'validation')
    ]
    
    print("Loading Train")
    train_data = MaestroDatasetWithWindowingInBeats(train_split, MAESTRO_ROOT,
                                                    mel_tx=log_mel,
                                                    n_mels=229,
                                                    subdivisions=12,
                                                    beats_per_window=8,
    hop_beats=8,
    hop_length=HOP_LENGTH,
    sr_target=16000,
    waveform_tx=None,
    mel_aug_prewarp=None,
    mel_aug_postwarp=None,
    target_transform=None,
    return_time_labels=False)
    print("Loading Val")
    val_data = MaestroDatasetWithWindowingInBeats(val_split, MAESTRO_ROOT,
                                                   mel_tx=log_mel,
                                                    n_mels=229,
                                                    subdivisions=12,
                                                    beats_per_window=8,
    hop_beats=8,                 # non-overlapping; set <8 for overlap
    hop_length=HOP_LENGTH,
    sr_target=16000,
    waveform_tx=None,
    mel_aug_prewarp=None,
    mel_aug_postwarp=None,
    target_transform=None,
    return_time_labels=False)
    
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST_TRAIN,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        collate_fn=collate_numeric
    )
    
    xb, yb = next(iter(train_loader))
    print("X:", xb.shape, xb.dtype)
    for k in ("on","off","frame"):
        v = yb[k]
        print(f"Y[{k}]:", type(v), v.dtype, v.shape, "sample=", v.flatten()[0:4])

    
    model = OnsetAndFrames(d_model=256, n_heads=8).to(device=DEVICE)
    
    opt = optim.Adam(model.parameters())
    criterion = OnsetsAndFramesLoss(pos_weight_on=None, pos_weight_off = None, pos_weight_frame=None)
    
    print("pw_on:", criterion.pw_on)
    print("pw_frm:", criterion.pw_frm)
    print("pw_off:", criterion.pw_off)

    
    for epoch in range(NUM_EPOCHS):
        tr, lr_track = train_one_epoch(model, train_loader, criterion, opt)
        print(f"[Onset and Frames] epoch {epoch+1}/{NUM_EPOCHS}  train={tr:.4f}  vlr_end={opt.param_groups[0]['lr']:.2e}")

if __name__ == "__main__" :
    main()