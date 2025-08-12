# utils/evaluation_tools.py
import os
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Optional: only needed if you want plots
try:
    import matplotlib.pyplot as plt
    from utils.display_midi import compare_piano_rolls_stacked
    _HAS_PLOTTING = True
except Exception:
    _HAS_PLOTTING = False


@torch.inference_mode()
def get_best_worst_examples(
    model: torch.nn.Module,
    val_loader,
    pos_weight: torch.Tensor = None,
    top_k: int = 3,
    device: str = "cuda",
    threshold: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run model over the validation loader and return the best/worst examples by loss.

    Returns:
        best:  list of dicts [{"loss": float, "pred": [T,128] float, "label": [T,128] float, "index": int}, ...]
        worst: same as above (highest losses)
    """
    model.eval()
    results = []

    # Prepare pos_weight
    pw = None
    if pos_weight is not None:
        pw = pos_weight.to(device)

    sample_index = 0  # running index across the whole val set

    for batch_idx, (x, y) in enumerate(tqdm(val_loader, desc="Evaluating", leave=False)):
        # y: [B, 128, T] -> [B, T, 128]; binarize
        y_bin = (y.permute(0, 2, 1) > 0).float().to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)

        logits = model(x)  # [B, T, 128]

        # elementwise BCE with optional pos_weight, then mean per sample
        loss_elems = F.binary_cross_entropy_with_logits(
            logits, y_bin, pos_weight=pw, reduction="none"
        )
        loss_per_sample = loss_elems.mean(dim=(1, 2))  # [B]

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for b in range(x.size(0)):
            results.append({
                "loss": float(loss_per_sample[b].item()),
                "pred": preds[b].detach().cpu(),     # [T, 128]
                "label": y_bin[b].detach().cpu(),    # [T, 128]
                "index": sample_index,
                "batch": batch_idx,
                "inbatch": b,
            })
            sample_index += 1

    # sort by loss
    results.sort(key=lambda d: d["loss"])
    best = results[:top_k]
    worst = results[-top_k:] if len(results) >= top_k else results[-len(results):]
    return best, worst


def save_best_worst_plots(
    best: List[Dict],
    worst: List[Dict],
    out_dir: str,
    model_name: str = "model",
) -> None:
    """
    Save side-by-side piano roll plots for the best/worst examples using your compare util.
    Assumes dicts from get_best_worst_examples(). Requires matplotlib + your util.
    """
    if not _HAS_PLOTTING:
        raise RuntimeError("Plotting not available. Install matplotlib and ensure utils.display_midi is importable.")

    os.makedirs(out_dir, exist_ok=True)

    def _save_one(item, tag, idx):
        # compare_piano_rolls_stacked expects [128, T]
        pred = item["pred"]
        lab  = item["label"]
        compare_piano_rolls_stacked(pred, lab, labels=("Predicted", "Target"))
        fn = os.path.join(out_dir, f"{model_name}_{tag}_{idx}_loss{item['loss']:.4f}.png")
        plt.savefig(fn, dpi=140, bbox_inches="tight")
        plt.close()

    for i, itm in enumerate(best, 1):
        _save_one(itm, "best", i)

    for i, itm in enumerate(reversed(worst), 1):
        _save_one(itm, "worst", i)
