import torch
from config import *
from data import make_loader, get_splits
from models.onset_and_frames import OnsetAndFrames
from models.loss import OnsetsAndFramesLoss
from tqdm import tqdm
import time

def build_model(model_name) :
    if  model_name == 'Onset and Frames' :
        return OnsetAndFrames(
        d_model=EMBED_DIM, 
        n_heads=N_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
        n_notes=N_NOTES,
        smooth_k=SMOOTH_K,
        detach_condition=DETACH_CONDITION
        )
    else :
        raise ValueError(f"Not a valid model name: {model_name}")
    
def save_model(model, tag: str):
    path = f"checkpoints/{tag}.pt"
    path_parent = "/".join(path.split("/")[:-1])
    if path_parent:
        import os; os.makedirs(path_parent, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)
    print(f"Saved model to {path}")

def train_one_epoch(
    model, 
    loader, 
    criterion, 
    optimizer, 
    LABEL_KEYS,
    scheduler=None, 
    step_per_batch=True, 
    device=DEVICE,
    return_stats=False,
):
    model.train()
    total_loss = 0.0
    lr_track = []

    total_data_time = 0.0
    total_compute_time = 0.0
    seen_samples = 0
    steps = 0

    # We'll manually iterate so we can time the data fetch separately
    iter_loader = iter(loader)

    # -------- warmup fetch (first batch) --------
    t_data_start = time.perf_counter()
    try:
        first_batch = next(iter_loader)
    except StopIteration:
        # empty loader edge case
        avg_loss = 0.0
        if return_stats:
            stats = {
                "data_time_s": 0.0,
                "compute_time_s": 0.0,
                "steps": 0,
                "avg_data_time_per_step_s": 0.0,
                "avg_compute_time_per_step_s": 0.0,
            }
            return avg_loss, lr_track, stats
        return avg_loss, lr_track

    t_data_end = time.perf_counter()
    total_data_time += (t_data_end - t_data_start)

    # prepare tqdm progress bar
    # we don't know len(iter_loader) easily because we manually iterate,
    # but we DO know len(loader) (number of batches). We'll use that for total.
    pbar = tqdm(total=len(loader), desc="train", leave=True)

    current_batch = first_batch

    while True:
        # batch comes from dataset as (mel, labels, meta)
        x, labels, meta = current_batch

        steps += 1
        t_compute_start = time.perf_counter()

        # ---------------- COMPUTE PHASE ----------------
        # Move input
        x = x.to(device, dtype=torch.float32, non_blocking=True)

        # Move/prepare label dict WITHOUT permuting batch dim
        y = {}
        for k in LABEL_KEYS:
            if k in labels:
                v = labels[k]
                # Accept [L,128] or [B,L,128]; normalize to [B,L,128]
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                y[k] = v.to(device, dtype=torch.float32, non_blocking=True)
                
        optimizer.zero_grad(set_to_none=True)

        out = model(x)                     # dict: {'on','off','frame',(optional 'vel')}
        loss, _metrics = criterion(out, y) # criterion expects dicts

        loss.backward()
        optimizer.step()

        if scheduler is not None and step_per_batch:
            scheduler.step()

        # bookkeeping
        batch_size = x.size(0)
        seen_samples += batch_size
        total_loss += loss.item() * batch_size
        lr_track.append(optimizer.param_groups[0]["lr"])

        # Optional one-time sanity print for shape debugging
        if steps == 1:
            print("x:", x.shape, x.dtype)
            for k in ("on", "off", "frame"):
                if k in y:
                    print(f"y[{k}]:", y[k].shape, y[k].dtype)
            for k in ("on", "off", "frame"):
                if k in out:
                    print(f"out[{k}]:", out[k].shape, out[k].dtype)

        t_compute_end = time.perf_counter()
        total_compute_time += (t_compute_end - t_compute_start)

        # ---------------- UPDATE PROGRESS BAR ----------------
        avg_loss_so_far = total_loss / max(1, seen_samples)
        avg_data_per_step = total_data_time / max(1, steps)
        avg_compute_per_step = total_compute_time / max(1, steps)
        current_lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "step": steps,
            "loss": f"{avg_loss_so_far:.4f}",
            "data_s/step": f"{avg_data_per_step:.4f}",
            "comp_s/step": f"{avg_compute_per_step:.4f}",
            "lr": f"{current_lr:.2e}",
        })
        pbar.update(1)

        # ---------------- NEXT BATCH FETCH PHASE ----------------
        t_data_start = time.perf_counter()
        try:
            current_batch = next(iter_loader)
        except StopIteration:
            break
        t_data_end = time.perf_counter()
        total_data_time += (t_data_end - t_data_start)

    # end while
    pbar.close()

    # Average loss over *seen* samples so partial epochs/cutoffs don't skew
    avg_loss = total_loss / max(1, seen_samples)

    if return_stats:
        stats = {
            "data_time_s": total_data_time,
            "compute_time_s": total_compute_time,
            "steps": steps,
            "avg_data_time_per_step_s": total_data_time / max(1, steps),
            "avg_compute_time_per_step_s": total_compute_time / max(1, steps),
        }
        return avg_loss, lr_track, stats

    return avg_loss, lr_track

@torch.no_grad()
def evaluate(model, loader, criterion, label_keys, device=DEVICE):
    model.eval()
    total = 0.0
    for x, labels in tqdm(loader, leave=False, desc="eval"):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = {}
        for k in label_keys:
            if k in labels:
                v = labels[k]
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                y[k] = v.to(device, dtype=torch.float32, non_blocking=True)
        out = model(x)
        loss, _ = criterion(out, y)
        total += loss.item() * x.size(0)
    return total / max(1, len(loader.dataset))

def train_model(model_name, normalization) :
    train_ds, val_ds = get_splits(
        transform=None,
        normalization=normalization
    )
    
    train_loader = make_loader(train_ds)
    val_loader = make_loader(val_ds)
    
    model = build_model(model_name).to(device=DEVICE)
    
    label_keys = ("on", "off", "frame", "vel")
    
    criterion = OnsetsAndFramesLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    best_val = float("inf")
    for epoch in range(NUM_EPOCHS):
        tr_loss, lr_track = train_one_epoch(
            model, train_loader, criterion, optimizer, label_keys
        )
        val_loss = evaluate(model, val_loader, criterion, label_keys)
        print(f"[{'norm' if normalization else 'plain'}] epoch {epoch+1}: "
              f"train={tr_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            tag = "onf_normalized_best" if normalization else "onf_plain_best"
            save_model(model, tag=tag)
    
    save_model(model)

def run_expirement(model_name) :
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    
    # Make the control group and then save it
    print('Training normalized model.')
    train_model(model_name, normalization=True)
    print('Finished training normalized model.')
    # Make the normalized model and then save it
    print('Training control model.')
    train_model(model_name, normalization = False)
    print('Finished training control model.')

def run_cost_estimation(
    model_name: str,
    log_path: str = "cost_estimate_log.txt"
):
    """
    Quick cost estimate / smoke test:
    - Builds two fresh models: normalized=True and normalized=False
    - For each: runs ONE truncated epoch (max_batches batches) on a small split
    - Measures data time vs compute time
    - Prints and appends a timing summary
    - Returns a dict of timings
    """

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    def _one_pass(normalization_flag: bool, label: str):
        """
        Helper to:
        - get small split
        - build model/optimizer/criterion fresh
        - run train_one_epoch once
        - return loss, stats, wallclock
        """
        # dataset / loader
        train_ds, _val_ds = get_splits(
            transform=None,
            normalization=normalization_flag,
            small=True,  # assumes your get_splits supports this
            subset_fraction=SUBSET_FRACTION
        )
        train_loader = make_loader(train_ds)

        # model + opt + loss fresh
        model = build_model(model_name).to(device=DEVICE)
        criterion = OnsetsAndFramesLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,
            weight_decay=1e-2
        )

        # time the whole "epoch"
        wall_t0 = time.perf_counter()
        avg_loss, lr_track, stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            LABEL_KEYS=("on", "off", "frame", "vel"),
            scheduler=None,
            step_per_batch=True,
            device=DEVICE,
            return_stats=True,    # <-- we need this arg in train_one_epoch
        )
        wall_t1 = time.perf_counter()

        wall = wall_t1 - wall_t0

        # bundle
        result = {
            "label": label,
            "avg_loss": avg_loss,
            "lr_final": lr_track[-1] if lr_track else None,
            "data_time_s": stats["data_time_s"],
            "compute_time_s": stats["compute_time_s"],
            "avg_data_time_per_step_s": stats["avg_data_time_per_step_s"],
            "avg_compute_time_per_step_s": stats["avg_compute_time_per_step_s"],
            "steps": stats["steps"],
            "wall_time_s": wall,
        }
        return result

    # run normalized
    print("Training normalized model (QUICK).")
    norm_result = _one_pass(True, "normalized")
    print("Finished training normalized model (QUICK).")

    # run control/plain
    print("Training control model (QUICK).")
    plain_result = _one_pass(False, "control")
    print("Finished training control model (QUICK).")

    overall_total_s = norm_result["wall_time_s"] + plain_result["wall_time_s"]

    # build nice summary
    summary = {
        "normalized_total_s": norm_result["wall_time_s"],
        "normalized_data_s": norm_result["data_time_s"],
        "normalized_compute_s": norm_result["compute_time_s"],

        "control_total_s": plain_result["wall_time_s"],
        "control_data_s": plain_result["data_time_s"],
        "control_compute_s": plain_result["compute_time_s"],

        "overall_total_s": overall_total_s,
    }

    scale_data   = 1.0 / SUBSET_FRACTION         # e.g. 25.0

    # projected time per model for full training job
    norm_full_training_s   = norm_result["wall_time_s"]   * scale_data * NUM_EPOCHS
    plain_full_training_s  = plain_result["wall_time_s"]  * scale_data * NUM_EPOCHS

    # combined (normalized + control, i.e. you train both variants)
    combined_full_training_s = norm_full_training_s + plain_full_training_s

    # also: per-epoch projection (full dataset, but just 1 epoch)
    norm_full_epoch_s  = norm_result["wall_time_s"]  * scale_data
    plain_full_epoch_s = plain_result["wall_time_s"] * scale_data


    # ==== PRINT HUMAN SUMMARY WITH PROJECTIONS ====
    print("\n===== RUN SUMMARY (COST-ESTIMATE) =====")
    print(f"Normalized pass (measured on subset x1 epoch):")
    print(f"  steps: {norm_result['steps']}")
    print(f"  loss: {norm_result['avg_loss']:.4f}")
    print(f"  wall (measured): {norm_result['wall_time_s']:.2f}s")
    print(f"    data:    {norm_result['data_time_s']:.2f}s "
          f"({norm_result['avg_data_time_per_step_s']:.4f}s/step)")
    print(f"    compute: {norm_result['compute_time_s']:.2f}s "
          f"({norm_result['avg_compute_time_per_step_s']:.4f}s/step)")

    print(f"\nControl pass (measured on subset x1 epoch):")
    print(f"  steps: {plain_result['steps']}")
    print(f"  loss: {plain_result['avg_loss']:.4f}")
    print(f"  wall (measured): {plain_result['wall_time_s']:.2f}s")
    print(f"    data:    {plain_result['data_time_s']:.2f}s "
          f"({plain_result['avg_data_time_per_step_s']:.4f}s/step)")
    print(f"    compute: {plain_result['compute_time_s']:.2f}s "
          f"({plain_result['avg_compute_time_per_step_s']:.4f}s/step)")

    print("\n--- Projected cost estimates ---")
    print(f"Per-model full epoch (100% data, 1 epoch):")
    print(f"  normalized: {norm_full_epoch_s/60:.2f} min")
    print(f"  control   : {plain_full_epoch_s/60:.2f} min")

    print(f"\nPer-model full training run ({NUM_EPOCHS} epochs on full data):")
    print(f"  normalized: {norm_full_training_s/3600:.2f} hr")
    print(f"  control   : {plain_full_training_s/3600:.2f} hr")

    print(f"\nBoth models total ({NUM_EPOCHS} epochs each):")
    print(f"  combined training time: {combined_full_training_s/3600:.2f} hr")

    print(f"\nOverall total wall time measured in smoke test (both models, subset, 1 epoch each): "
          f"{overall_total_s:.2f}s")
    print("=======================================\n")

    # ==== LOG FILE OUTPUT (MATCHES ABOVE) ====
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write("===== RUN SUMMARY (COST-ESTIMATE) =====\n")
            f.write(f"model_name: {model_name}\n")

            f.write("\nNormalized pass (measured on subset ×1 epoch):\n")
            f.write(f"  steps: {norm_result['steps']}\n")
            f.write(f"  loss: {norm_result['avg_loss']:.6f}\n")
            f.write(f"  wall_time_s: {norm_result['wall_time_s']:.6f}\n")
            f.write(f"  data_time_s: {norm_result['data_time_s']:.6f}\n")
            f.write(f"  compute_time_s: {norm_result['compute_time_s']:.6f}\n")
            f.write(f"  avg_data_time_per_step_s: {norm_result['avg_data_time_per_step_s']:.6f}\n")
            f.write(f"  avg_compute_time_per_step_s: {norm_result['avg_compute_time_per_step_s']:.6f}\n")

            f.write("\nControl pass (measured on subset ×1 epoch):\n")
            f.write(f"  steps: {plain_result['steps']}\n")
            f.write(f"  loss: {plain_result['avg_loss']:.6f}\n")
            f.write(f"  wall_time_s: {plain_result['wall_time_s']:.6f}\n")
            f.write(f"  data_time_s: {plain_result['data_time_s']:.6f}\n")
            f.write(f"  compute_time_s: {plain_result['compute_time_s']:.6f}\n")
            f.write(f"  avg_data_time_per_step_s: {plain_result['avg_data_time_per_step_s']:.6f}\n")
            f.write(f"  avg_compute_time_per_step_s: {plain_result['avg_compute_time_per_step_s']:.6f}\n")

            f.write("\n--- Projected cost estimates ---\n")
            f.write(f"Per-model full epoch (100% data, 1 epoch):\n")
            f.write(f"  normalized: {norm_full_epoch_s/60:.2f} min\n")
            f.write(f"  control:    {plain_full_epoch_s/60:.2f} min\n")

            f.write(f"\nPer-model full training run ({NUM_EPOCHS} epochs on full data):\n")
            f.write(f"  normalized: {norm_full_training_s/3600:.2f} hr\n")
            f.write(f"  control:    {plain_full_training_s/3600:.2f} hr\n")

            f.write(f"\nBoth models total ({NUM_EPOCHS} epochs each):\n")
            f.write(f"  combined training time: {combined_full_training_s/3600:.2f} hr\n")

            f.write(f"\nOverall total wall time measured in smoke test "
                f"(both models, subset, 1 epoch each): {overall_total_s:.2f}s\n")
            f.write("=======================================\n\n")

    return summary
    

if __name__ == "__main__" :
    model_name = "Onset and Frames"
    run_cost_estimation(model_name)