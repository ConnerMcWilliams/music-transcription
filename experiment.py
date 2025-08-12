import torch
import torch.nn as nn
from tqdm import tqdm
from models.basic_CNN import BasicAMTCNN
from config import (DEVICE, NUM_EPOCHS, RESULTS_DIR,
                    SAMPLES_PER_CLIP, FRAMES_PER_CLIP)
from schedulers import make_optimizer, make_scheduler
from losses import make_criterion, estimate_pos_weight

def build_model(model_cfg):
    t = model_cfg["type"]
    if t == "BasicAMTCNN":
        return BasicAMTCNN(SAMPLES_PER_CLIP, FRAMES_PER_CLIP)
    else:
        raise ValueError(f"Unknown model type: {t}")

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None, step_per_batch=True):
    model.train()
    total = 0.0
    lr_track = []
    for x, y in tqdm(loader, leave=False, desc="train"):
        y = (y.permute(0,2,1) > 0).float().to(DEVICE, non_blocking=True)
        x = x.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        assert out.shape == y.shape, (out.shape, y.shape)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if scheduler is not None and step_per_batch:
            scheduler.step()

        lr_track.append(optimizer.param_groups[0]["lr"])
        total += loss.item()
    return total / max(1, len(loader)), lr_track

@torch.inference_mode()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total = 0.0
    for x, y in tqdm(loader, leave=False, desc="valid"):
        y = (y.permute(0,2,1) > 0).float().to(DEVICE, non_blocking=True)
        x = x.to(DEVICE, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        total += loss.item()
    return total / max(1, len(loader))

def run_experiment(train_loader, val_loader, variant, pos_weight_vec=None):
    # Build model
    model = build_model(variant["model"]).to(DEVICE)

    # Optional: bias init from base rate
    if pos_weight_vec is not None and hasattr(model, "fc") and hasattr(model.fc, "bias"):
        with torch.no_grad():
            p_k = 1.0 / (1.0 + pos_weight_vec)
            bias = torch.log(p_k / (1.0 - p_k))
            model.fc.bias.copy_(bias.to(DEVICE))

    # Optimizer & scheduler
    opt = make_optimizer(variant["optimizer"]["type"], model.parameters(), variant["optimizer"]["lr"])
    scheduler, step_per_batch = make_scheduler(variant["scheduler"]["type"], opt, len(train_loader))

    crit = make_criterion(pos_weight_vec)

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
