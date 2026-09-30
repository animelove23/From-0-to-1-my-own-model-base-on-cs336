"""Convert UTF-8 text into little-endian uint16 tokens for training.

Run with --input, --output, --vocab and --merges. Use --resume-checkpoint
for an existing partial output; otherwise the output must not already exist.
"""

import argparse
import hashlib
import json
import os
import pickle

import numpy as np

from cs336_basics.tokenizer_class import Tokenizer


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resume(args):
    with open(args.resume_checkpoint, encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    source = checkpoint["source"]
    output = checkpoint["output"]
    tokenizer = checkpoint["tokenizer"]
    expected = (
        (args.input, source), (args.output, output),
        (args.vocab, {"sha256": tokenizer["vocab_sha256"]}),
        (args.merges, {"sha256": tokenizer["merges_sha256"]}),
    )
    for path, metadata in expected:
        if "size_bytes" in metadata and os.path.getsize(path) != metadata["size_bytes"]:
            raise ValueError(f"Size differs from checkpoint: {path}")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Hash differs from checkpoint: {path}")
    if output.get("dtype") != "uint16" or not output.get("verified_decoding_equals_source_prefix"):
        raise ValueError("Checkpoint does not describe a verified uint16 output")
    if not checkpoint["resume"].get("at_line_boundary"):
        raise ValueError("Checkpoint is not at a complete line boundary")
    return source["resume_byte_offset"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 source text")
    parser.add_argument("--output", required=True, help="Output .bin file")
    parser.add_argument("--vocab", required=True, help="Pickled byte vocabulary")
    parser.add_argument("--merges", required=True, help="Pickled BPE merges")
    parser.add_argument("--resume-checkpoint", help="Checkpoint for the existing output")
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument("--buffer-tokens", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.buffer_tokens <= 0:
        parser.error("--buffer-tokens must be positive")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("Input and output must be different files")
    offset = verify_resume(args) if args.resume_checkpoint else 0
    with open(args.vocab, "rb") as stream:
        vocab = pickle.load(stream)
    with open(args.merges, "rb") as stream:
        merges = pickle.load(stream)
    tokenizer = Tokenizer(vocab, merges, [args.special_token])
    buffer = []
    # 'xb' prevents overwriting a previous conversion; 'ab' is allowed only
    # after verifying the source, tokenizer and existing output hashes.
    mode = "ab" if args.resume_checkpoint else "xb"
    with open(args.input, "rb") as source, open(args.output, mode) as output:
        source.seek(offset)
        for raw_line in source:
            ids = tokenizer.encode(raw_line.decode("utf-8"))
            if any(token < 0 or token > 65535 for token in ids):
                raise ValueError("Token id exceeds uint16 range")
            buffer.extend(ids)
            if len(buffer) >= args.buffer_tokens:
                np.asarray(buffer, dtype="<u2").tofile(output)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype="<u2").tofile(output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()