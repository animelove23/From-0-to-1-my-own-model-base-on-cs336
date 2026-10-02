"""Plot the 64M / 128M training run directly from its published CSV.

From assignment1-basics:
    uv run --no-sync python scalling_exp/plot_64m_best_checkpoint.py

The blue band shows the 10th–90th percentiles of single-batch training loss
within a rolling window of 101 logged points. The solid blue line is the
rolling mean. Validation loss is plotted as measured, without smoothing.
No checkpoint weights or private dataset are required to reproduce the plot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_RUN = Path(__file__).resolve().parent / "64m_model" / "64m_128mtoken"
WINDOW = 101
BLUE = "#2563eb"
ORANGE = "#ea580c"
RED = "#dc2626"
INK = "#14243a"
MUTED = "#64748b"


def read_run(run: Path):
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
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
    best_step = int(result["best_validation_step"])
    best_loss = float(result["best_validation_loss"])
    matched = validation[validation[:, 0] == best_step]
    if len(matched) != 1 or not math.isclose(matched[0, 2], best_loss, abs_tol=1e-8):
        raise ValueError("Best checkpoint does not match the validation log")
    if int(validation[-1, 0]) != int(result["final_step"]):
        raise ValueError("Final step does not match the validation log")
    if not math.isclose(validation[-1, 2], result["final_validation_loss"], abs_tol=1e-8):
        raise ValueError("Final validation loss does not match result.json")
    return result, train, validation, matched[0, 1]


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
    ax.set_facecolor("#fbfdff")
    ax.grid(color="#dbe5f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#b8c6d9")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot(run: Path):
    result, train, validation, best_tokens = read_run(run)
    mean, lower, upper = rolling_summary(train[:, 2], WINDOW)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "font.size": 10,
    })
    fig = plt.figure(figsize=(14.4, 6.8), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.75, 1), left=0.065, right=0.975,
        top=0.70, bottom=0.18, wspace=0.13,
    )
    full, zoom = (fig.add_subplot(grid[0, i]) for i in range(2))
    for ax in (full, zoom):
        style_axis(ax)
        ax.fill_between(train[:, 1], lower, upper, color=BLUE, alpha=0.12, linewidth=0)
        ax.plot(train[:, 1], mean, color=BLUE, linewidth=1.8)
        ax.plot(validation[:, 1], validation[:, 2], color=ORANGE, linewidth=2.1)
        ax.scatter([best_tokens], [result["best_validation_loss"]],
                   s=42, color=RED, edgecolor="white", linewidth=0.9, zorder=7)
        ax.set_xlabel("Training tokens (millions)", color=INK, labelpad=9)

    full.set_xlim(0, 128.5)
    full.set_ylim(1.2, 9.6)
    full.set_xticks((0, 32, 64, 96, 128))
    full.set_ylabel("Cross-entropy loss (nats)", color=INK, labelpad=9)
    full.set_title("A  |  Complete training run", loc="left", color=INK, weight="bold", pad=12)

    zoom.set_xlim(96, 128.5)
    zoom.set_ylim(1.36, 2.02)
    zoom.set_xticks((96, 104, 112, 120, 128))
    zoom.set_title("B  |  Final 32M tokens", loc="left", color=INK, weight="bold", pad=12)
    zoom.annotate(
        f"BEST CHECKPOINT\nstep {int(result['best_validation_step']):,}  ·  val {result['best_validation_loss']:.4f}",
        xy=(best_tokens, result["best_validation_loss"]), xycoords="data",
        xytext=(0.06, 0.94), textcoords="axes fraction",
        ha="left", va="top", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.65", facecolor="white", edgecolor="#dbe5f0"),
    )

    fig.text(0.065, 0.92, "64M model  /  128M training tokens",
             fontsize=20, weight="bold", color=INK)
    fig.text(0.065, 0.865,
             "Training loss is aggregated across logged mini-batches; validation loss is shown as measured.",
             fontsize=10, color=MUTED)
    handles = [
        Line2D([0], [0], color=BLUE, lw=2, label=f"Train · {WINDOW}-point rolling mean"),
        Patch(facecolor=BLUE, alpha=0.14, label="Train · rolling 10th–90th percentile"),
        Line2D([0], [0], color=ORANGE, lw=2, label="Validation · measured"),
        Line2D([0], [0], color=RED, marker="o", markersize=6, lw=0, label="Best checkpoint"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.795),
               ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.065, 0.075,
             f"Best: step {int(result['best_validation_step']):,} · val {result['best_validation_loss']:.4f}"
             f"     |     Final: step {int(result['final_step']):,} · val {result['final_validation_loss']:.4f}"
             "     |     Source: training_log.csv",
             fontsize=9.5, color=MUTED)

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
