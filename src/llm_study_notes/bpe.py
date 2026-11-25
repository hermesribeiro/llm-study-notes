import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

import regex as re
from tqdm import tqdm, trange


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_chunks = 160
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

        pending = set()
        frequency_table = {}
        with ProcessPoolExecutor(max_workers=16) as executor:
            for start, end in tqdm(pairwise(boundaries), total=num_chunks, desc="Pre-tokenizing"):
                # Free capacity before loading another chunk.
                if len(pending) == 16:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    while done:
                        frequency_table = combine_tables(frequency_table, done.pop().result())

                f.seek(start)
                raw_text = f.read(end - start).decode("utf-8", errors="ignore")
                pending.add(executor.submit(pre_tokenize, raw_text, special_tokens))

            # Consume the remaining results.
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                while done:
                    frequency_table = combine_tables(frequency_table, done.pop().result())

    tokens = [bytes([x]) for x in range(256)]
    tokens.extend([x.encode("utf-8") for x in special_tokens])

    vocab, merges = merge_step(vocab_size, frequency_table, tokens)

    return vocab, merges


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

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    # Applying pretokenizer
    pattern = "|".join([s.replace("|", "\\|") for s in special_tokens])
    corpus = re.split(pattern, text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    corpus_pretoken = [re.finditer(PAT, c) for c in corpus]

    # Computing word frequency
    frequency_table = {}
    for document in corpus_pretoken:
        for word in document:
            word_bytes = tuple(bytes([x]) for x in word.group().encode("utf-8"))
            frequency_table[word_bytes] = frequency_table.get(word_bytes, 0) + 1

    return frequency_table


def combine_tables(dict1: dict, dict2: dict):
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in dict1.keys() | dict2.keys()}


def merge_step(
    vocab_size: int, frequency_table: dict[tuple[bytes, ...], int], tokens: list[bytes]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []
    words = list(frequency_table.keys())
    table = list(frequency_table.values())
    word2idx = dict(zip(frequency_table.keys(), range(len(words))))

    # Getting the most frequent byte pair
    pairs = {}
    index = {}
    for idx in trange(len(words), desc="Getting pairs"):
        word = words[idx]
        count = table[idx]
        for pair in pairwise(word):
            pairs[pair] = pairs.get(pair, 0) + count
            index[pair] = index.get(pair, set()) | {idx}

    with tqdm(total=vocab_size, desc="Merging iteratively") as pbar:
        while len(tokens) < vocab_size and len(pairs) < len(words):
            pbar.update(1)
            byte_pair = max(pairs, key=lambda pair: (pairs[pair],) + pair)
            merges.append(byte_pair)
            merged = byte_pair[0] + byte_pair[1]
            tokens.append(merged)

            # Merging the words and recomputing pairs and index
            for idx in index[byte_pair]:
                word = words[idx]
                new_word = merge_word(word, byte_pair, merged)
                words[idx] = new_word
                del word2idx[word]
                word2idx[new_word] = idx
                count = table[idx]
                for idi, piece in enumerate(new_word):
                    if piece != merged:
                        continue

                    if idi < len(new_word) - 1:
                        if new_word[idi + 1] != merged:
                            adjacent = (byte_pair[1], new_word[idi + 1])
                            new_pair = (merged, new_word[idi + 1])
                            pairs[adjacent] -= count
                            pairs[new_pair] = pairs.get(new_pair, 0) + count
                            index[new_pair] = index.get(new_pair, set()) | {idx}
                            clean_dict(pairs, adjacent)

                        else:
                            adjacent = (byte_pair[1], byte_pair[0])
                            new_pair = (merged, merged)
                            pairs[adjacent] -= count
                            pairs[new_pair] = pairs.get(new_pair, 0) + count
                            index[new_pair] = index.get(new_pair, set()) | {idx}
                            clean_dict(pairs, adjacent)

                    if idi > 0:
                        if new_word[idi - 1] != merged:
                            adjacent = (new_word[idi - 1], byte_pair[0])
                            new_pair = (new_word[idi - 1], merged)
                            pairs[adjacent] -= count
                            pairs[new_pair] = pairs.get(new_pair, 0) + count
                            index[new_pair] = index.get(new_pair, set()) | {idx}
                            clean_dict(pairs, adjacent)
                        else:
                            pass

            del pairs[byte_pair]

    vocab = {k: v for k, v in enumerate(tokens)}

    return vocab, merges


def merge_word(word: tuple[bytes, ...], byte_pair: tuple[bytes, bytes], merged: bytes):
    idi = 0
    new_word = ()
    while idi < len(word):
        if word[idi : idi + 2] == byte_pair:
            new_word += (merged,)
            idi += 2
        else:
            new_word += (word[idi],)
            idi += 1

    return new_word


def clean_dict(pairs: dict, pair: tuple[bytes, bytes]):
    if pairs[pair] == 0 or pairs[pair] == {}:
        del pairs[pair]
    return pairs


def bytes_handler(obj):
    if isinstance(obj, bytes):
        return "".join(chr(b).replace("\\", "\\\\") if 32 <= b <= 126 else f"\\x{b:02x}" for b in obj)
    else:
        return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE and save the vocabulary and merges beside the corpus.")
    parser.add_argument("corpus_path", type=Path, help="Path to the training corpus")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Target vocabulary size (default: 10000)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        help="Reserve an additional special token (repeatable). <|endoftext|> is always included.",
    )
    args = parser.parse_args()

    special_tokens = args.special_token or []
    end_of_text = "<|endoftext|>"
    if end_of_text not in special_tokens:
        special_tokens.insert(0, end_of_text)

    corpus_path = args.corpus_path
    vocab_path = corpus_path.with_name(f"{corpus_path.stem}_vocab{corpus_path.suffix}")
    merges_path = corpus_path.with_name(f"{corpus_path.stem}_merges{corpus_path.suffix}")

    start_time = time.perf_counter()
    vocab, merges = train_bpe(corpus_path, args.vocab_size, special_tokens)

    with open(vocab_path, "w") as f:
        json.dump(vocab, f, default=bytes_handler)

    with open(merges_path, "wb") as f:
        f.writelines(left + b"<|sep|>" + right + b"<|eol|>\n" for left, right in merges)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"{execution_time=}")


if __name__ == "__main__":
    main()
