"""Train a Transformer using a private JSON configuration.

Run: python -m cs336_basics.public.training --config CONFIG.json
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from cs336_basics.AdamW import AdamW
from cs336_basics.Transformer import Transformer
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import data_loading
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import learning_rate_schedule

REQUIRED_KEYS = (
    "train_data_path", "checkpoint_dir", "vocab_size", "context_length",
    "d_model", "num_layers", "num_heads", "d_ff", "rope_theta",
    "batch_size", "max_steps", "max_lr", "min_lr", "warmup_steps",
    "cosine_steps", "weight_decay", "beta1", "beta2", "eps",
    "max_grad_norm", "log_interval", "eval_interval", "eval_batches",
    "save_interval", "seed",
)


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError("Missing configuration keys: " + ", ".join(missing))
    return config


def load_tokens(path):
    path = str(path)
    if path.endswith(".npy"):
        data = np.load(path, mmap_mode="r")
    else:
        data = np.memmap(path, dtype=np.uint16, mode="r")
    if data.ndim != 1:
        raise ValueError("Expected a one-dimensional token array")
    return data


def loss_for_batch(model, x, y):
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def evaluate(model, data, config, device):
    model.eval()
    losses = []
    for _ in range(config["eval_batches"]):
        x, y = data_loading(data, config["batch_size"], config["context_length"], device)
        losses.append(loss_for_batch(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Private JSON training configuration")
    args = parser.parse_args()
    config = read_config(args.config)
    if any(config[key] <= 0 for key in ("log_interval", "eval_interval", "eval_batches", "save_interval")):
        raise ValueError("Intervals and eval_batches must be positive")

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    tokens = load_tokens(config["train_data_path"])
    if config.get("val_data_path"):
        train_tokens = tokens
        val_tokens = load_tokens(config["val_data_path"])
    else:
        fraction = config.get("val_fraction")
        if fraction is None or not 0 < fraction < 1:
            raise ValueError("Provide val_data_path or val_fraction between 0 and 1")
        split = int(len(tokens) * (1 - fraction))
        train_tokens, val_tokens = tokens[:split], tokens[split:]
    if min(len(train_tokens), len(val_tokens)) <= config["context_length"]:
        raise ValueError("Training and validation arrays must exceed context_length")

    model = Transformer(
        config["d_ff"], config["d_model"], config["num_heads"],
        config["vocab_size"], config["context_length"], config["num_layers"],
        True, config["rope_theta"],
    ).to(device)
    parameters = list(model.parameters())
    optimizer = AdamW(
        parameters, lambd=config["weight_decay"], lr=config["max_lr"],
        betas=(config["beta1"], config["beta2"]), eps=config["eps"],
    )
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(("step", "train_loss", "val_loss", "val_ppl", "lr"))

    start_step = 0
    if config.get("resume_checkpoint"):
        model, optimizer, start_step = load_checkpoint(
            config["resume_checkpoint"], model, optimizer
        )
    model.train()
    for step in range(start_step + 1, config["max_steps"] + 1):
        lr = learning_rate_schedule(
            step, config["max_lr"], config["min_lr"],
            config["warmup_steps"], config["cosine_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = data_loading(train_tokens, config["batch_size"], config["context_length"], device)
        loss = loss_for_batch(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(parameters, config["max_grad_norm"])
        optimizer.step()
        if step == 1 or step % config["log_interval"] == 0:
            print(f"step {step} | train loss {loss.item():.4f} | lr {lr:.6e}")
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow((step, loss.item(), "", "", lr))
        if step % config["eval_interval"] == 0:
            val_loss = evaluate(model, val_tokens, config, device)
            print(f"step {step} | val loss {val_loss:.4f}")
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow((step, "", val_loss, math.exp(val_loss), lr))
        if step % config["save_interval"] == 0:
            save_checkpoint(model, optimizer, step, checkpoint_dir / f"step_{step}.pt")
    save_checkpoint(model, optimizer, config["max_steps"], checkpoint_dir / "final.pt")


if __name__ == "__main__":
    main()