import torch
import os, pickle
from config import (SEED, DEVICE, MODEL_VARIANTS, RESULTS_DIR, POSW_PATH)
from data import get_splits, make_loader
from dataset.transforms import log_mel
from losses import estimate_pos_weight
from experiment import run_experiment
from plotting import save_histories_csv, save_lr_csv, plot_losses, plot_lrs

def _safe_torch_load(path):
    """Load with map_location='cpu' and try weights_only=True if supported."""
    kwargs = {"map_location": "cpu"}
    # weights_only is available in newer PyTorch; try it, fall back if not
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)

def load_or_compute_pos_weight(train_loader, path=POSW_PATH, max_batches=100, force=False):
    should_compute = force or (not os.path.isfile(path)) or os.path.getsize(path) == 0
    if not should_compute:
        try:
            w = _safe_torch_load(path)
            if not isinstance(w, torch.Tensor):
                raise ValueError("Cached pos_weight is not a tensor.")
            print(f"Loaded pos_weight from {path}")
        except (pickle.UnpicklingError, EOFError, RuntimeError, ValueError) as e:
            print(f"Failed to load cached pos_weight ({e}). Recomputing...")
            should_compute = True

    if should_compute:
        print("Computing Positive Weights...")
        w = estimate_pos_weight(train_loader, max_batches=max_batches)
        w = w.clamp_(1.0, 50.0).cpu()
        # atomic-ish save: write tmp then replace
        tmp = path + ".tmp"
        torch.save(w, tmp)
        os.replace(tmp, path)
        print(f"Saved pos_weight to {path}")

    print("pos_weight stats:", w.min().item(), w.max().item(), w.mean().item())
    return w


def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    train_ds, val_ds = get_splits(log_mel)
    print("Loading Train Data...")
    train_loader = make_loader(train_ds, train=True)
    print("Train Data Loaded!")
    print("Loading Eval Data...")
    val_loader   = make_loader(val_ds,   train=False)
    print("Eval Data Loaded!")

    pos_w_k = load_or_compute_pos_weight(train_loader=train_loader)

    histories = {}
    for variant in MODEL_VARIANTS:
        print(f"\n=== Running {variant['name']} ===")
        h = run_experiment(train_loader, val_loader, variant, pos_weight_vec=pos_w_k)
        histories[variant["name"]] = h

    save_histories_csv(histories)
    save_lr_csv(histories)
    plot_losses(histories)
    plot_lrs(histories)

if __name__ == "__main__":
    main()
