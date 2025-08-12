import os
import csv
import matplotlib.pyplot as plt
from config import RESULTS_DIR, NUM_EPOCHS

def save_histories_csv(histories, filename="loss_curves.csv"):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch", "train_loss", "val_loss"])
        for name, h in histories.items():
            for ep, (tr, va) in enumerate(zip(h["train"], h["val"]), start=1):
                w.writerow([name, ep, tr, va])
    print(f"Saved {path}")

def save_lr_csv(histories, filename="lr_schedules.csv"):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "batch_index", "lr"])
        for name, h in histories.items():
            for i, lr in enumerate(h["lr"], start=1):
                w.writerow([name, i, lr])
    print(f"Saved {path}")

def plot_losses(histories, filename="loss_curves.png"):
    plt.figure(figsize=(11, 7))
    for name, h in histories.items():
        epochs = range(1, NUM_EPOCHS + 1)
        plt.plot(epochs, h["train"], label=f"{name} — train")
        plt.plot(epochs, h["val"], linestyle="--", label=f"{name} — val")
    plt.xlabel("Epoch"); plt.ylabel("Loss (BCEWithLogits)")
    plt.title("Training & Validation Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=160)
    print(f"Saved {path}")

def plot_lrs(histories, filename="lr_schedules.png"):
    plt.figure(figsize=(11, 5))
    for name, h in histories.items():
        if h["lr"]:
            plt.plot(range(1, len(h["lr"]) + 1), h["lr"], label=name)
    plt.xlabel("Batch step"); plt.ylabel("Learning Rate")
    plt.title("LR Schedules")
    plt.grid(True); plt.legend(); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=160)
    print(f"Saved {path}")
