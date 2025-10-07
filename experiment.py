import torch
import torch.nn as nn
from tqdm import tqdm
from models.basic_CNN import BasicAMTCNN
from models.basic_transformer import BasicTransformerAMT
from config import (DEVICE, NUM_EPOCHS, RESULTS_DIR,
                    SAMPLES_PER_CLIP, FRAMES_PER_CLIP, SUBDIVISIONS_PER_BEAT,
                    BEATS_PER_CLIP, N_MELS)
from schedulers import make_optimizer, make_scheduler
from losses import make_criterion, estimate_pos_weight
from models.onset_and_frames import OnsetAndFrames

def build_model(model_cfg):
    t = model_cfg["type"]
    if t == "OnsetAndFrames" :
        return OnsetAndFrames(d_model=256, n_heads=8)
    else:
        raise ValueError(f"Unknown model type: {t}")
    
LABEL_KEYS = ("on", "off", "frame", "vel")

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

    avg_loss = total / max(1, len(loader.dataset))
    return avg_loss, lr_track

@torch.inference_mode()
def eval_one_epoch(model, loader, criterion, device=DEVICE):
    model.eval()
    total = 0.0
    for step, (x, labels) in tqdm(loader, leave=False, desc="valid"):
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
                
        out = model(x)
        loss = criterion(out, y)
        total += loss.item()
    return total / max(1, len(loader))

def run_experiment(train_loader, val_loader, variant, pos_weight_vec=None):
    # Build model
    model = build_model(variant["model"]).to(DEVICE)

    # Optimizer & scheduler
    opt = make_optimizer(variant["optimizer"]["type"], model.parameters(), variant["optimizer"]["lr"])
    scheduler, step_per_batch = make_scheduler(variant["scheduler"]["type"], opt, len(train_loader))

    crit = make_criterion()

    train_hist, val_hist, lr_hist = [], [], []

    for epoch in range(NUM_EPOCHS):
        tr, lr_track = train_one_epoch(model, train_loader, crit, opt, scheduler, step_per_batch)
        va = eval_one_epoch(model, val_loader, crit)
        if scheduler is not None and not step_per_batch:
            scheduler.step(va)  # e.g. plateau

        train_hist.append(tr)
        val_hist.append(va)
        lr_hist.extend(lr_track)
        print(f"[{variant['name']}] epoch {epoch+1}/{NUM_EPOCHS}  train={tr:.4f}  val={va:.4f}  lr_end={opt.param_groups[0]['lr']:.2e}")

    # Save weights
    torch.save(model.state_dict(), f"{RESULTS_DIR}/{variant['name']}.pth")
    return {"train": train_hist, "val": val_hist, "lr": lr_hist}
