import multiprocessing
import os
import regex as re
from collections import Counter
from typing import BinaryIO

def pretokenization(start,end, path):
    with open(path, "rb") as corpus:
        corpus.seek(start)
        corpus_content = corpus.read(end-start)
        corpus_content = corpus_content.decode("utf-8")
        segments = corpus_content.split("<|endoftext|>")
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        counter = Counter()
        for segment in segments:
            for match in re.finditer(PAT, segment):
                counter[match.group()]+=1
        return counter
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def pretoken_and_statistic(input_path: str):
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        pair = []
        for i,j in zip(boundaries[:-1], boundaries[1:]):
            pair.append((i,j, input_path))
        with multiprocessing.Pool(num_processes) as pool:
            result = pool.starmap(pretokenization, pair)
            counter = Counter()
            for chunk in result:
                counter.update(chunk)
            new_counter = Counter()
            for k,v in counter.items():
                btuple = (bytes([e]) for e in k.encode("utf-8"))
                new_counter[tuple(btuple)] = v
    return new_counter


