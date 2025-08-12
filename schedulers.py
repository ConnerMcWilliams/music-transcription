import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau
from config import NUM_EPOCHS, WEIGHT_DECAY

def make_optimizer(name, params, lr):
    if name.lower() == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    elif name.lower() == "adam":
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def onecycle_scheduler(optimizer, steps_per_epoch, max_lr):
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e3
    ), True  # step_per_batch

def cosine_warmup_scheduler(optimizer, total_iters, warmup_iters):
    warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    cos  = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters)
    return SequentialLR(optimizer, [warm, cos], milestones=[warmup_iters]), True

def plateau_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6), False

def make_scheduler(tag, optimizer, steps_per_epoch):
    if tag == "onecycle":
        return onecycle_scheduler(optimizer, steps_per_epoch, optimizer.param_groups[0]['lr'])
    elif tag == "cosine_warmup":
        total_iters  = steps_per_epoch * NUM_EPOCHS
        warmup_iters = max(steps_per_epoch // 20, 1)   # ~5% warmup
        return cosine_warmup_scheduler(optimizer, total_iters, warmup_iters)
    elif tag == "plateau":
        return plateau_scheduler(optimizer)
    elif tag in (None, "none"):
        return None, False
    else:
        raise ValueError(f"Unknown scheduler tag: {tag}")
