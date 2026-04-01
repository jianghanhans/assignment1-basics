"""
tokenizer_experiments.py

问题 tokenizer_experiments (a)(b)(c)(d)

运行方式:
  python tokenizer_experiments.py
"""
from __future__ import annotations

import pathlib
import random
import time
import numpy as np

from tests.adapters import Tokenizer, load_bpe_vocab, load_bpe_merges

# ── paths ──────────────────────────────────────────────────────────────────────
TS_TRAIN  = pathlib.Path("data/TinyStoriesV2-GPT4-train.txt")
TS_VALID  = pathlib.Path("data/TinyStoriesV2-GPT4-valid.txt")
OWT_TRAIN = pathlib.Path("data/owt_train.txt")
OWT_VALID = pathlib.Path("data/owt_valid.txt")

TS_VOCAB_PATH   = pathlib.Path("output/tinystories_bpe/vocab.json")
TS_MERGES_PATH  = pathlib.Path("output/tinystories_bpe/merges.txt")
OWT_VOCAB_PATH  = pathlib.Path("output/owt_bpe/vocab.json")
OWT_MERGES_PATH = pathlib.Path("output/owt_bpe/merges.txt")

OUT_DIR = pathlib.Path("output/tokenized")
SPECIAL_TOKENS = ["<|endoftext|>"]
RANDOM_SEED = 42


def load_tokenizer(vocab_path, merges_path):
    vocab  = load_bpe_vocab(vocab_path)
    merges = load_bpe_merges(merges_path)
    return Tokenizer(vocab, merges, SPECIAL_TOKENS)


def sample_documents(filepath: pathlib.Path, n: int, seed: int = RANDOM_SEED) -> list[str]:
    """Sample n documents split by <|endoftext|>."""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    docs = [d.strip() for d in text.split("<|endoftext|>") if d.strip()]
    rng = random.Random(seed)
    return rng.sample(docs, min(n, len(docs)))


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """bytes per token."""
    total_bytes  = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens if total_tokens else 0.0


def encode_file_to_npy(
    tokenizer: Tokenizer,
    src: pathlib.Path,
    dst: pathlib.Path,
) -> None:
    """Encode an entire file to a uint16 numpy array and save to dst."""
    ids: list[int] = []
    with open(src, encoding="utf-8", errors="ignore") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
    arr = np.array(ids, dtype=np.uint16)
    np.save(dst, arr)
    print(f"  Saved {len(arr):,} tokens → {dst}  ({dst.stat().st_size/1024**2:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ts_tok_available  = TS_VOCAB_PATH.exists()  and TS_MERGES_PATH.exists()
    owt_tok_available = OWT_VOCAB_PATH.exists() and OWT_MERGES_PATH.exists()

    if not ts_tok_available:
        print("TinyStories tokenizer not found. Run train_bpe_tinystories.py first.")
    if not owt_tok_available:
        print("OWT tokenizer not found. Run train_bpe_owt.py first.")

    # ── load tokenizers ────────────────────────────────────────────────────────
    ts_tok  = load_tokenizer(TS_VOCAB_PATH,  TS_MERGES_PATH)  if ts_tok_available  else None
    owt_tok = load_tokenizer(OWT_VOCAB_PATH, OWT_MERGES_PATH) if owt_tok_available else None

    # ── sample 10 docs from each dataset ──────────────────────────────────────
    ts_docs  = sample_documents(TS_TRAIN,  10) if TS_TRAIN.exists()  else []
    owt_docs = sample_documents(OWT_TRAIN, 10) if OWT_TRAIN.exists() else []

    # ══════════════════════════════════════════════════════════════════════════
    # (a) Compression ratio on matching datasets
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("(a) Compression ratio (bytes/token)")
    print("="*60)

    if ts_tok and ts_docs:
        cr = compression_ratio(ts_tok, ts_docs)
        print(f"  TinyStories tokenizer on TinyStories docs : {cr:.3f} bytes/token")

    if owt_tok and owt_docs:
        cr = compression_ratio(owt_tok, owt_docs)
        print(f"  OWT tokenizer on OWT docs                 : {cr:.3f} bytes/token")

    # ══════════════════════════════════════════════════════════════════════════
    # (b) Cross-tokenizer: TinyStories tokenizer on OWT docs
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("(b) Cross-tokenizer compression")
    print("="*60)

    if ts_tok and owt_docs:
        cr_ts_on_owt = compression_ratio(ts_tok, owt_docs)
        print(f"  TinyStories tokenizer on OWT docs         : {cr_ts_on_owt:.3f} bytes/token")

    if owt_tok and owt_docs:
        cr_owt_on_owt = compression_ratio(owt_tok, owt_docs)
        print(f"  OWT tokenizer on OWT docs                 : {cr_owt_on_owt:.3f} bytes/token")

    if ts_tok and owt_docs and owt_tok:
        ratio = cr_ts_on_owt / cr_owt_on_owt
        print(f"  Ratio (TS/OWT tokenizer on OWT)           : {ratio:.3f}x")
        print("  → TinyStories tokenizer produces more tokens on OWT text")
        print("    (lower compression) because its vocab was trained on simpler")
        print("    children's stories and lacks web-text subwords.")

    # ══════════════════════════════════════════════════════════════════════════
    # (c) Throughput estimate
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("(c) Throughput estimate")
    print("="*60)

    bench_tok = ts_tok or owt_tok
    bench_docs = ts_docs or owt_docs
    if bench_tok and bench_docs:
        bench_text = "\n".join(bench_docs)
        bench_bytes = len(bench_text.encode("utf-8"))
        # warm-up
        bench_tok.encode(bench_text[:1000])
        # timed run
        t0 = time.perf_counter()
        for _ in range(5):
            bench_tok.encode(bench_text)
        elapsed = (time.perf_counter() - t0) / 5
        throughput_mb_s = bench_bytes / elapsed / 1024**2
        throughput_gb_s = throughput_mb_s / 1024
        pile_gb = 825
        pile_hours = pile_gb / (throughput_gb_s * 3600)
        print(f"  Benchmark text size : {bench_bytes/1024:.1f} KB")
        print(f"  Throughput          : {throughput_mb_s:.1f} MB/s")
        print(f"  Pile (825 GB) ETA   : {pile_hours:.1f} hours")

    # ══════════════════════════════════════════════════════════════════════════
    # (d) Encode full datasets to uint16 numpy arrays
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("(d) Encode full datasets → uint16 numpy arrays")
    print("="*60)
    print("  Why uint16: vocab sizes are 10K and 32K, both < 65536 (2^16),")
    print("  so uint16 is the smallest integer type that fits all token IDs,")
    print("  using 2 bytes/token vs 4 bytes for uint32 — halves storage and")
    print("  memory bandwidth when loading training batches.")
    print()

    if ts_tok:
        for split, path in [("train", TS_TRAIN), ("valid", TS_VALID)]:
            if path.exists():
                dst = OUT_DIR / f"tinystories_{split}.npy"
                print(f"  Encoding TinyStories {split} ...")
                t0 = time.perf_counter()
                encode_file_to_npy(ts_tok, path, dst)
                print(f"  Time: {time.perf_counter()-t0:.1f}s")

    if owt_tok:
        for split, path in [("train", OWT_TRAIN), ("valid", OWT_VALID)]:
            if path.exists():
                dst = OUT_DIR / f"owt_{split}.npy"
                print(f"  Encoding OWT {split} ...")
                t0 = time.perf_counter()
                encode_file_to_npy(owt_tok, path, dst)
                print(f"  Time: {time.perf_counter()-t0:.1f}s")

    print("\nDone.")
