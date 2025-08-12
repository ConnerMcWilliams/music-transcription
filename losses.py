import torch
import torch.nn as nn
from config import DEVICE

@torch.no_grad()
def estimate_pos_weight(loader, max_batches=None):
    pos = None
    tot = 0
    for i, (_, y) in enumerate(loader):
        y = (y > 0).float()               # [B, 128, T]
        per_note_pos = y.sum(dim=(0, 2))  # [128]
        per_note_tot = y.shape[0] * y.shape[2]  # B*T per note (scalar)
        if pos is None:
            pos = per_note_pos
            tot = per_note_tot
        else:
            pos += per_note_pos
            tot += per_note_tot
        if max_batches is not None and (i + 1) >= max_batches:
            break

    eps = 1e-6
    p_k = (pos / (tot + eps)).clamp(min=eps)  # [128]
    w_k = ((1 - p_k) / p_k)                   # [128]
    return w_k



def make_criterion(pos_weight_vec=None, clip=(1.0, 50.0)):
    if pos_weight_vec is None:
        return nn.BCEWithLogitsLoss()
    w = pos_weight_vec.clone()
    if clip is not None:
        w.clamp_(clip[0], clip[1])
    return nn.BCEWithLogitsLoss(pos_weight=w.to(DEVICE))
