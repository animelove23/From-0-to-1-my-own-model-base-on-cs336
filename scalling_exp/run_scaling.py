"""Independent 4 x 4 model/token-budget experiments; existing source files are untouched.

Prepare: python scalling_exp/run_scaling.py --prepare
Run:     python scalling_exp/run_scaling.py --run
Plot:    python scalling_exp/plot_scaling.py
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from cs336_basics.public.training import loss_for_batch
from cs336_basics.Transformer import Transformer
from cs336_basics.AdamW import AdamW
from cs336_basics.data_loading import data_loading
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from plot_scaling import plot_results, run_directory

MODELS = {
    "8M": dict(d_model=256, num_layers=4, num_heads=4, d_ff=768),
    "16M": dict(d_model=384, num_layers=4, num_heads=6, d_ff=1152),
    "32M": dict(d_model=512, num_layers=6, num_heads=8, d_ff=1536),
    "64M": dict(d_model=640, num_layers=10, num_heads=10, d_ff=1920),
}
STEPS = {16: 7813, 32: 15625, 64: 31250, 128: 62500}
STOP = False
LOG_FIELDS = ("step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr", "elapsed_seconds")
DEFAULT_MANIFEST = json.loads((ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))
DEFAULTS = DEFAULT_MANIFEST["settings"]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def data_description(path, size=None):
    path = Path(path).resolve()
    size = path.stat().st_size if size is None else size
    if size % 2 or size <= 0 or path.stat().st_size < size:
        raise ValueError(f"Invalid uint16 file size: {path}")
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"Data was truncated: {path}")
            digest.update(block)
            remaining -= len(block)
    return {"path": str(path), "size_bytes": size, "tokens": size // 2, "sha256": digest.hexdigest()}


def settings(args):
    return dict(
        batch_size=8, context_length=256, vocab_size=DEFAULTS["vocab_size"],
        rope_theta=DEFAULTS["rope_theta"], max_lr=DEFAULTS["max_lr"], min_lr=DEFAULTS["min_lr"],
        warmup_ratio=DEFAULTS["warmup_ratio"],
        weight_decay=DEFAULTS["weight_decay"], beta1=DEFAULTS["beta1"], beta2=DEFAULTS["beta2"],
        eps=DEFAULTS["eps"], max_grad_norm=DEFAULTS["max_grad_norm"],
        seed=args.seed, validation_seed=args.seed + 1,
        eval_batches=args.eval_batches, eval_interval=args.eval_interval,
        log_interval=DEFAULTS["log_interval"], save_interval=args.save_interval,
        val_fraction=DEFAULTS["val_fraction"], initialization="original component initialization, including FFN Xavier normal",
    )


def parameter_count(shape, vocab_size):
    d, layers, ff = shape["d_model"], shape["num_layers"], shape["d_ff"]
    return 2 * vocab_size * d + layers * (4 * d * d + 3 * d * ff + 2 * d) + d


def prepare(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment_manifest.json"
    common = settings(args)
    names = ["Transformer.py", "transformer_block.py", "multihead_self_attention.py", "RoPE.py",
             "RMSNorm.py", "FFN.py", "AdamW.py", "cross_entropy.py", "data_loading.py",
             "gradient_clipping.py", "learning_rate_schedule.py", "linear_model.py", "embadding.py",
             "scaled_dot_product_attention.py", "softmax.py"]
    hashes = {name: hashlib.sha256((PROJECT / "cs336_basics" / name).read_bytes()).hexdigest() for name in names}
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != common or any(manifest["source_hashes"].get(name) != digest for name, digest in hashes.items()):
            raise ValueError("Existing experiment settings/source changed. Use the original settings or a new --output-dir.")
        frozen_data = data_description(args.data, manifest["data"]["size_bytes"])
        if frozen_data["sha256"] != manifest["data"]["sha256"]:
            raise ValueError("Frozen training data prefix changed")
        expected_val = manifest["validation_data"]
        if bool(args.val_data) != bool(expected_val):
            raise ValueError("Validation data setting differs from manifest")
        if expected_val:
            frozen_val = data_description(args.val_data, expected_val["size_bytes"])
            if frozen_val["sha256"] != expected_val["sha256"]:
                raise ValueError("Frozen validation data changed")
    else:
        data = data_description(args.data)
        validation = data_description(args.val_data) if args.val_data else None
        split = data["tokens"] if validation else int(data["tokens"] * (1 - common["val_fraction"]))
        val_count = validation["tokens"] if validation else data["tokens"] - split
        if min(split, val_count) <= common["context_length"]:
            raise ValueError("Not enough tokens for train/validation split")
        manifest = dict(settings=common, data=data, validation_data=validation, train_tokens=split,
                        validation_tokens=val_count, source_hashes=hashes,
                        comparison="16 independently initialized runs; same frozen split and fixed validation batches",
                        metric="validation loss at exact final step; not best checkpoint loss",
                        note="16M budget rounds to 16,001,024 processed tokens; sampling is with replacement")
        atomic_json(path, manifest)
    signature = fingerprint(manifest)
    for model, shape in MODELS.items():
        for budget, steps in STEPS.items():
            run = run_directory(root, model, budget)
            (run / "checkpoints").mkdir(parents=True, exist_ok=True)
            config = dict(model=model, nominal_tokens_million=budget, max_steps=steps,
                          actual_tokens=steps * 8 * 256, shape=shape, settings=common,
                          warmup_steps=max(1, int(steps * common["warmup_ratio"])),
                          parameter_count=parameter_count(shape, common["vocab_size"]),
                          manifest_fingerprint=signature)
            config_path = run / "config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError(f"Run config changed: {config_path}")
            atomic_json(config_path, config)
    runtime_manifest = json.loads(json.dumps(manifest))
    runtime_manifest["data"]["path"] = str(args.data.resolve())
    if args.val_data:
        runtime_manifest["validation_data"]["path"] = str(args.val_data.resolve())
    return root, runtime_manifest


def load_data(manifest):
    data = manifest["data"]
    full = np.memmap(data["path"], dtype="<u2", mode="r", shape=(data["tokens"],))
    train = full[:manifest["train_tokens"]]
    if manifest["validation_data"]:
        val = manifest["validation_data"]
        validation = np.memmap(val["path"], dtype="<u2", mode="r", shape=(val["tokens"],))
    else:
        validation = full[manifest["train_tokens"]:]
    return train, validation


def build_model(config, device):
    c, s = config["shape"], config["settings"]
    torch.manual_seed(s["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s["seed"])
    model = Transformer(c["d_ff"], c["d_model"], c["num_heads"], s["vocab_size"],
                        s["context_length"], c["num_layers"], True, s["rope_theta"])
    assert sum(p.numel() for p in model.parameters()) == config["parameter_count"]
    return model.to(device)


@torch.no_grad()
def evaluate(model, data, s, device):
    rng = np.random.get_state()
    np.random.seed(s["validation_seed"])
    model.eval()
    total = 0.0
    try:
        for _ in range(s["eval_batches"]):
            x, y = data_loading(data, s["batch_size"], s["context_length"], device)
            total += loss_for_batch(model(x), y).item()
    finally:
        np.random.set_state(rng)
        model.train()
    return total / s["eval_batches"]


def request_stop(signum, frame):
    global STOP
    STOP = True


def train_one(run, config, train, val, device):
    global STOP
    result_path = run / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") == "completed" and result["config_fingerprint"] == fingerprint(config):
            print(f"Skip completed: {run.name}", flush=True)
            return True
        raise ValueError(f"Incompatible existing result: {result_path}")
    logger = logging.getLogger(run.name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.FileHandler(run / "console.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    s = config["settings"]
    model = build_model(config, device)
    params = list(model.parameters())
    optimizer = AdamW(params, lambd=s["weight_decay"], lr=s["max_lr"],
                      betas=(s["beta1"], s["beta2"]), eps=s["eps"])
    np.random.seed(s["seed"])
    latest = run / "checkpoints" / "latest.pt"
    step = 0
    best_loss, best_step, prior_seconds = float("inf"), 0, 0.0
    restore_path = latest if latest.exists() else run / 'checkpoints' / 'final.pt'
    if restore_path.exists():
        saved = torch.load(restore_path, map_location="cpu", weights_only=False)
        if saved["config_fingerprint"] != fingerprint(config):
            raise ValueError(f"Checkpoint configuration differs: {latest}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, best_loss, best_step = saved["iteration"], saved["best_loss"], saved["best_step"]
        prior_seconds = saved["elapsed_seconds"]
    log_path = run / "training_log.csv"
    # Remove only rows after the restored checkpoint; interrupted steps are rerun.
    previous = []
    if log_path.exists():
        with log_path.open(newline="") as stream:
            previous = [row for row in csv.DictReader(stream) if int(row["step"]) <= step]
    with log_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(previous)
    started = time.monotonic()

    def save_latest():
        atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=step,
                              numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                              best_loss=best_loss, best_step=best_step,
                              elapsed_seconds=prior_seconds + time.monotonic() - started,
                              config_fingerprint=fingerprint(config), config=config), latest)

    logger.info("%s: parameters=%s, target steps=%s, processed tokens=%s, resume step=%s",
                run.name, config["parameter_count"], config["max_steps"], config["actual_tokens"], step)
    model.train()
    final_loss = None
    try:
        while step < config["max_steps"]:
            next_step = step + 1
            lr = learning_rate_schedule(next_step, s["max_lr"], s["min_lr"],
                                        config["warmup_steps"], config["max_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = data_loading(train, s["batch_size"], s["context_length"], device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {next_step}")
            loss.backward()
            gradient_clipping(params, s["max_grad_norm"])
            optimizer.step()
            step = next_step
            val_loss = None
            if step % s["eval_interval"] == 0 or step == config["max_steps"]:
                val_loss = evaluate(model, val, s, device)
                if not math.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite validation loss at step {step}")
                if val_loss < best_loss:
                    best_loss, best_step = val_loss, step
                    atomic_torch_save(dict(model=model.state_dict(), iteration=step,
                                           validation_loss=val_loss, config=config), run / "checkpoints" / "best.pt")
                if step == config["max_steps"]:
                    final_loss = val_loss
            if step == 1 or step % s["log_interval"] == 0 or val_loss is not None or STOP:
                elapsed = prior_seconds + time.monotonic() - started
                with log_path.open("a", newline="") as stream:
                    csv.writer(stream).writerow((step, step*2048, loss.item(),
                        val_loss if val_loss is not None else "",
                        math.exp(val_loss) if val_loss is not None and val_loss < 700 else "", lr, elapsed))
                logger.info("step=%d train=%.5f val=%s lr=%.6g", step, loss.item(), val_loss, lr)
            if step % s["save_interval"] == 0 or STOP:
                save_latest()
            if STOP:
                logger.info("Stopped after step %d; rerun the same command to resume.", step)
                return False
        if final_loss is None:
            final_loss = evaluate(model, val, s, device)
        save_latest()
        os.replace(latest, run / "checkpoints" / "final.pt")
        result = dict(status="completed", model=config["model"],
                      nominal_tokens_million=config["nominal_tokens_million"],
                      actual_tokens=config["actual_tokens"], final_step=step,
                      final_validation_loss=final_loss, best_validation_loss=best_loss,
                      best_validation_step=best_step, parameter_count=config["parameter_count"],
                      config_fingerprint=fingerprint(config),
                      elapsed_seconds=prior_seconds + time.monotonic() - started)
        atomic_json(result_path, result)
        logger.info("Completed: final validation loss %.6f", final_loss)
        return True
    except Exception:
        logger.exception("Run failed; latest periodic checkpoint can be resumed.")
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true", help="Create configs and pending plots; no training")
    action.add_argument("--run", action="store_true", help="Run selected independent experiments sequentially")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/train.bin" if (PROJECT / "data/train.bin").exists() else Path(DEFAULT_MANIFEST["data"]["path"]))
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--budgets", nargs="+", type=int, choices=list(STEPS), default=list(STEPS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--eval-batches", type=int, default=DEFAULTS["eval_batches"])
    parser.add_argument("--eval-interval", type=int, default=DEFAULTS["eval_interval"])
    parser.add_argument("--save-interval", type=int, default=DEFAULTS["save_interval"])
    args = parser.parse_args()
    if min(args.eval_batches, args.eval_interval, args.save_interval) < 1:
        parser.error("Intervals and eval-batches must be positive")
    for model in args.models:
        for budget in args.budgets:
            print(f"{model:>3} / {budget:>3}M tokens: {STEPS[budget]:>5} steps, "
                  f"{STEPS[budget]*2048:,} actual tokens, "
                  f"{parameter_count(MODELS[model], DEFAULTS["vocab_size"]):,} parameters")
    if not args.prepare and not args.run:
        print("Plan only. Use --prepare to create configs or --run to start training.")
        return
    root, manifest = prepare(args)
    plot_results(root)
    if args.prepare:
        print("Prepared 16 configs. No training started.")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; choose --device cpu explicitly")
    train, val = load_data(manifest)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    for model in args.models:
        for budget in args.budgets:
            if STOP:
                return
            run = run_directory(root, model, budget)
            config = json.loads((run / "config.json").read_text())
            complete = train_one(run, config, train, val, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            plot_results(root)
            if not complete:
                return


if __name__ == "__main__":
    main()