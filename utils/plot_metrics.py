import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    repo_root = Path(__file__).resolve().parent
    default_metrics = repo_root / "results" / "metrics.jsonl"
    default_output = repo_root / "results" / "metrics_summary.png"

    parser = argparse.ArgumentParser(
        description="Plot training metrics from a JSONL metrics log."
    )
    parser.add_argument(
        "metrics_path",
        nargs="?",
        default=str(default_metrics),
        help="Path to metrics JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Path to output PNG file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving.",
    )
    return parser.parse_args()


def load_metrics(metrics_path):
    records = []

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {metrics_path}: {exc}"
                ) from exc

            if "epoch" not in record:
                continue
            records.append(record)

    if not records:
        raise ValueError(f"No valid metric rows found in {metrics_path}")

    records.sort(key=lambda item: item["epoch"])
    return records


def series(records, key):
    xs = []
    ys = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        xs.append(record["epoch"])
        ys.append(value)
    return xs, ys


def plot_if_present(ax, records, key, label, **kwargs):
    xs, ys = series(records, key)
    if not xs:
        return False
    ax.plot(xs, ys, label=label, **kwargs)
    return True


def best_record(records, key, mode="min"):
    candidates = [record for record in records if key in record]
    if not candidates:
        return None
    if mode == "min":
        return min(candidates, key=lambda item: item[key])
    return max(candidates, key=lambda item: item[key])


def build_figure(records, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Training Metrics Summary", fontsize=16)

    loss_ax = axes[0, 0]
    has_loss = False
    has_loss |= plot_if_present(
        loss_ax, records, "train_loss", "Train Loss", color="tab:blue", linewidth=2
    )
    has_loss |= plot_if_present(
        loss_ax, records, "val_loss", "Val Loss", color="tab:orange", linewidth=2
    )
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    if has_loss:
        loss_ax.legend()

    f1_ax = axes[0, 1]
    has_f1 = False
    has_f1 |= plot_if_present(
        f1_ax, records, "onset_f1", "Onset F1", color="tab:green", linewidth=2
    )
    has_f1 |= plot_if_present(
        f1_ax, records, "frame_f1", "Frame F1", color="tab:red", linewidth=2
    )
    f1_ax.set_title("F1 Scores")
    f1_ax.set_xlabel("Epoch")
    f1_ax.set_ylabel("F1")
    f1_ax.grid(True, alpha=0.3)
    if has_f1:
        f1_ax.legend()

    lr_ax = axes[1, 0]
    has_lr = plot_if_present(
        lr_ax, records, "lr", "Learning Rate", color="tab:purple", linewidth=2
    )
    lr_ax.set_title("Learning Rate")
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("LR")
    lr_ax.grid(True, alpha=0.3)
    if has_lr:
        _, lr_values = series(records, "lr")
        if all(value > 0 for value in lr_values):
            lr_ax.set_yscale("log")
        lr_ax.legend()

    pr_ax = axes[1, 1]
    has_pr = False
    has_pr |= plot_if_present(
        pr_ax, records, "onset_prec", "Onset Precision", color="tab:blue", linewidth=2
    )
    has_pr |= plot_if_present(
        pr_ax, records, "onset_rec", "Onset Recall", color="tab:cyan", linewidth=2
    )
    has_pr |= plot_if_present(
        pr_ax, records, "frame_prec", "Frame Precision", color="tab:orange", linewidth=2
    )
    has_pr |= plot_if_present(
        pr_ax, records, "frame_rec", "Frame Recall", color="tab:brown", linewidth=2
    )
    pr_ax.set_title("Precision / Recall")
    pr_ax.set_xlabel("Epoch")
    pr_ax.set_ylabel("Score")
    pr_ax.grid(True, alpha=0.3)
    if has_pr:
        pr_ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def print_summary(records, output_path):
    print(f"Saved metrics summary to: {output_path}")

    best_val = best_record(records, "val_loss", mode="min")
    if best_val is not None:
        print(
            f"Best val_loss: epoch {best_val['epoch']} -> {best_val['val_loss']:.6f}"
        )

    best_frame_f1 = best_record(records, "frame_f1", mode="max")
    if best_frame_f1 is not None:
        print(
            f"Best frame_f1: epoch {best_frame_f1['epoch']} -> "
            f"{best_frame_f1['frame_f1']:.6f}"
        )



def main():
    args = parse_args()
    metrics_path = Path(args.metrics_path).resolve()
    output_path = Path(args.output).resolve()

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    records = load_metrics(metrics_path)
    fig = build_figure(records, output_path)
    print_summary(records, output_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
