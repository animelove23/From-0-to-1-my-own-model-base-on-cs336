"""Plot the 64M / 128M training run directly from its published CSV.

Run from the repository root:
    uv run --no-project --with matplotlib --with numpy python scalling_exp/plot_64m_best_checkpoint.py

The blue band shows the 10th–90th percentiles of single-batch training loss
within a rolling window of 101 logged points. The solid blue line is the
rolling mean. Validation loss is plotted as measured, without smoothing.
The published training_log.csv is the only data source.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RUN = Path(__file__).resolve().parent / "64m_model" / "64m_128mtoken"
WINDOW = 101
BLUE = "#2563eb"
ORANGE = "#ea580c"
INK = "#14243a"
MUTED = "#64748b"


def read_run(run: Path):
    train, validation = [], []
    with (run / "training_log.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "tokens_seen", "train_loss", "val_loss"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing required columns in {run / 'training_log.csv'}")
        for row in reader:
            step, tokens = int(row["step"]), int(row["tokens_seen"])
            if row["train_loss"]:
                train.append((step, tokens / 1e6, float(row["train_loss"])))
            if row["val_loss"]:
                validation.append((step, tokens / 1e6, float(row["val_loss"])))
    train = np.asarray(train, dtype=float)
    validation = np.asarray(validation, dtype=float)
    if train.ndim != 2 or validation.ndim != 2:
        raise ValueError("Training or validation series is empty")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Training log contains non-finite values")
    if np.any(np.diff(train[:, 0]) <= 0) or np.any(np.diff(validation[:, 0]) <= 0):
        raise ValueError("Training log steps must increase strictly")
    return train, validation


def rolling_summary(loss: np.ndarray, window: int):
    """Centered rolling mean and empirical 10th–90th percentile envelope."""
    radius = window // 2
    mean = np.empty(len(loss))
    lower = np.empty(len(loss))
    upper = np.empty(len(loss))
    for i in range(len(loss)):
        segment = loss[max(0, i - radius):min(len(loss), i + radius + 1)]
        mean[i] = segment.mean()
        lower[i], upper[i] = np.quantile(segment, (0.1, 0.9))
    return mean, lower, upper


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(color="#e4e9ef", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#b8c6d9")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot(run: Path):
    train, validation = read_run(run)
    mean, lower, upper = rolling_summary(train[:, 2], WINDOW)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(11.4, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.16)
    style_axis(ax)
    ax.fill_between(train[:, 1], lower, upper, color=BLUE, alpha=0.13,
                    linewidth=0, label="Training 10th–90th percentile")
    ax.plot(train[:, 1], mean, color=BLUE, linewidth=1.8,
            label=f"Training {WINDOW}-point mean")
    ax.plot(validation[:, 1], validation[:, 2], color=ORANGE,
            linewidth=1.9, label="Validation")
    ax.set_xlim(0, 128.5)
    ax.set_ylim(1.2, 9.6)
    ax.set_xticks((0, 32, 64, 96, 128))
    ax.set_xlabel("Training tokens (millions)", color=INK, labelpad=8)
    ax.set_ylabel("Cross-entropy loss (nats)", color=INK, labelpad=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.085, 0.985),
               ncol=3, frameon=False, fontsize=9, labelcolor=INK)

    outputs = []
    for suffix in (".png",):
        output = run / f"best_checkpoint_training_curve{suffix}"
        fig.savefig(output, dpi=190, facecolor="white")
        outputs.append(output)
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    run = parser.parse_args().run_dir.resolve()
    for path in plot(run):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
