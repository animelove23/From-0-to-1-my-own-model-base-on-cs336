"""Plot final validation losses for the 4 x 4 scaling experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("8M", "16M", "32M", "64M")
BUDGETS = (16, 32, 64, 128)
COLORS = ("#2563eb", "#059669", "#ea580c", "#7c3aed")


def run_directory(root, model, budget):
    return root / f"{model.lower()}_model" / f"{model.lower()}_{budget}mtoken"


def plot_results(root):
    root = Path(root)
    values = np.full((4, 4), np.nan)
    rows = []
    for i, model in enumerate(MODELS):
        for j, budget in enumerate(BUDGETS):
            path = run_directory(root, model, budget) / "result.json"
            row = {"model": model, "budget_million_tokens": budget, "status": "pending",
                   "actual_tokens": "", "final_validation_loss": "", "parameter_count": ""}
            if path.exists():
                result = json.loads(path.read_text())
                if result.get("status") == "completed":
                    loss = float(result["final_validation_loss"])
                    if not np.isfinite(loss):
                        raise ValueError(f"Non-finite final loss: {path}")
                    values[i, j] = loss
                    row.update(status="completed", actual_tokens=result["actual_tokens"],
                               final_validation_loss=loss, parameter_count=result["parameter_count"])
            rows.append(row)
    with (root / "results_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed = int(np.isfinite(values).sum())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
    for i, model in enumerate(MODELS):
        ax.plot(BUDGETS, values[i], marker="o", lw=2, ms=6, color=COLORS[i], label=model)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [f"{x}M" for x in BUDGETS])
    ax.set_xlabel("Training tokens (processed, not unique)")
    ax.set_ylabel("Final validation loss (nats)")
    ax.set_title("Training tokens vs. validation loss", loc="left", weight="bold", pad=15)
    ax.legend(title="Model size", frameon=False)
    ax.grid(alpha=0.18)
    if not completed:
        ax.text(0.5, 0.5, "Awaiting completed experiments\nNo measured losses yet",
                transform=ax.transAxes, ha="center", va="center", color="#64748b")
        ax.set_yticks([])
    fig.supxlabel(f"{completed}/16 completed | independent runs | final-step evaluation", fontsize=9, color="#64748b")
    for extension in ("png", "pdf"):
        fig.savefig(root / f"tokens_vs_validation_loss.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#e2e8f0")
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max())) if completed else (0, 1)
    if low == high:
        low, high = low - 0.05, high + 0.05
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=low, vmax=high, aspect="auto")
    for i in range(4):
        for j in range(4):
            v = values[i, j]
            label = f"{v:.4f}" if np.isfinite(v) else "Pending"
            color = "white" if np.isfinite(v) and (v-low)/(high-low) > 0.55 else "#0f172a"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks(range(4), [f"{x}M" for x in BUDGETS])
    ax.set_yticks(range(4), MODELS)
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Model size")
    ax.set_title("4 x 4 final validation loss", loc="left", weight="bold", pad=15)
    if completed:
        fig.colorbar(im, ax=ax, label="Validation loss (lower is better)")
    fig.supxlabel(f"{completed}/16 completed | Pending cells are not results", fontsize=9, color="#64748b")
    for extension in ("png", "pdf"):
        fig.savefig(root / f"validation_loss_heatmap.{extension}", dpi=180)
    plt.close(fig)
    print(f"Updated plots: {root} ({completed}/16 completed)")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    plot_results(parser.parse_args().output_dir)