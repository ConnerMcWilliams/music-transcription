import torch, os
from torch.utils.data import DataLoader
from models.basic_CNN import BasicAMTCNN
from data import get_splits, make_loader
from dataset.transforms import log_mel
from losses import make_criterion
from config import (SAMPLES_PER_CLIP, FRAMES_PER_CLIP, DEVICE, RESULTS_DIR, N_MELS)
from utils.best_and_worst import save_best_worst_plots, get_best_worst_examples

MODEL_NAME = "Transformer_OneCycle_max3e-3"

def load_pos_weight_from_results():
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith("pos_weight") and fname.endswith(".pt"):
            return torch.load(os.path.join(RESULTS_DIR, fname), map_location="cpu")
    return None  # fine: the utility can handle None

def load_model_by_name(name: str):
    model = BasicAMTCNN(SAMPLES_PER_CLIP, n_frames=FRAMES_PER_CLIP, n_mels=N_MELS).to(DEVICE)
    state = torch.load(os.path.join(RESULTS_DIR, f"{name}.pth"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.inference_mode()
def evaluate_model(weights_path, pos_weight_path=None, threshold=0.5):
    train_ds, val_ds = get_splits(log_mel)
    val_loader = make_loader(val_ds, train=False)

    model = BasicAMTCNN(SAMPLES_PER_CLIP, n_frames=FRAMES_PER_CLIP, n_mels=N_MELS).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    pos_w = torch.load(pos_weight_path, map_location=DEVICE) if pos_weight_path else None
    criterion = make_criterion(pos_w)

    total = 0.0
    for x, y in val_loader:
        y = (y.permute(0,2,1) > 0).float().to(DEVICE, non_blocking=True)
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item()
    print(f"Validation loss: {total / max(1, len(val_loader)):.4f}")
    
def main():
    # Load validation data
    _, val_ds = get_splits(log_mel)
    val_loader = make_loader(val_ds, train=False)

    # Load pos_weight (if exists)
    pos_w = load_pos_weight_from_results()

    # Load model
    weights_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}.pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No weights found at {weights_path}")

    model = load_model_by_name(MODEL_NAME)

    # Evaluate best/worst
    best, worst = get_best_worst_examples(
        model=model,
        val_loader=val_loader,
        pos_weight=pos_w,
        top_k=3,
        device=DEVICE,
        threshold=0.5,
    )

    # Save plots
    out_dir = os.path.join(RESULTS_DIR, "eval")
    save_best_worst_plots(best, worst, out_dir=out_dir, model_name=MODEL_NAME)

    # Print results
    print("\nBest 3:")
    for i, d in enumerate(best, 1):
        print(f"  #{i}: idx={d['index']}, loss={d['loss']:.4f}")
    print("Worst 3:")
    for i, d in enumerate(reversed(worst), 1):
        print(f"  #{i}: idx={d['index']}, loss={d['loss']:.4f}")

if __name__ == "__main__":
    main()