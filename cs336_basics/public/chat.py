"""Interactive text generation using paths and model settings from a private JSON file."""

import argparse
import json
import pickle

import torch

from cs336_basics.Transformer import Transformer
from cs336_basics.tokenizer_class import Tokenizer

REQUIRED_KEYS = (
    "checkpoint_path", "vocab_path", "merges_path", "vocab_size", "context_length",
    "d_model", "num_layers", "num_heads", "d_ff", "rope_theta", "max_new_tokens",
)


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError("Missing configuration keys: " + ", ".join(missing))
    return config


def sample_token(logits, temperature, top_p):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    probabilities = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
    keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    return sorted_ids[torch.multinomial(sorted_probs, 1)].item()


def generate(prompt, model, tokenizer, config, device):
    ids = tokenizer.encode(prompt)
    if not ids:
        return ""
    prompt_length = len(ids)
    stop_token = config.get("stop_token", "<|endoftext|>")
    stop_id = tokenizer.encode(stop_token)[0]
    for _ in range(config["max_new_tokens"]):
        if len(ids) >= config["context_length"]:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(x)[0, -1]
        token = sample_token(logits, config.get("temperature", 1.0), config.get("top_p", 1.0))
        if token == stop_id:
            break
        ids.append(token)
    return b"".join(tokenizer.vocab[token] for token in ids[prompt_length:]).decode(
        "utf-8", errors="replace"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Private JSON inference configuration")
    args = parser.parse_args()
    config = read_config(args.config)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    with open(config["vocab_path"], "rb") as stream:
        vocab = pickle.load(stream)
    with open(config["merges_path"], "rb") as stream:
        merges = pickle.load(stream)
    tokenizer = Tokenizer(vocab, merges, [config.get("stop_token", "<|endoftext|>")])
    model = Transformer(
        config["d_ff"], config["d_model"], config["num_heads"],
        config["vocab_size"], config["context_length"], config["num_layers"],
        True, config["rope_theta"],
    ).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    while True:
        try:
            prompt = input("Prompt (quit to exit): ")
        except EOFError:
            break
        if prompt.strip() == "quit":
            break
        print(generate(prompt, model, tokenizer, config, device))


if __name__ == "__main__":
    main()