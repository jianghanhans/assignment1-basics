"""
train_bpe_owt.py

问题 train_bpe_expts_owt (a)(b):
  在 OpenWebText 上训练字节级 BPE 分词器，vocab_size=32000。
  序列化词表和 merges，打印训练时间、内存峰值、最长 token。
  同时加载 TinyStories 的结果做对比。

运行方式:
  python train_bpe_owt.py
"""
import time
import tracemalloc
import pathlib
import multiprocessing

from tests.adapters import run_train_bpe, save_bpe_vocab, save_bpe_merges, load_bpe_vocab, load_bpe_merges

DATA_PATH = pathlib.Path("data/owt_train.txt")
OUT_DIR = pathlib.Path("output/owt_bpe")
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_WORKERS = multiprocessing.cpu_count()  # 56 on this server

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE on {DATA_PATH} (vocab_size={VOCAB_SIZE}, workers={NUM_WORKERS}) ...")
    tracemalloc.start()
    t0 = time.time()

    vocab, merges = run_train_bpe(
        input_path=DATA_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        num_workers=NUM_WORKERS,
    )

    elapsed = time.time() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    save_bpe_vocab(vocab, OUT_DIR / "vocab.json")
    save_bpe_merges(merges, OUT_DIR / "merges.txt")
    print(f"Saved vocab to {OUT_DIR / 'vocab.json'}")
    print(f"Saved merges to {OUT_DIR / 'merges.txt'}")

    longest_token = max(vocab.values(), key=len)
    print(f"\n=== OWT Results ===")
    print(f"Training time : {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
    print(f"Peak memory   : {peak_bytes / 1024**3:.2f} GB")
    print(f"Vocab size    : {len(vocab)}")
    print(f"Merges count  : {len(merges)}")
    print(f"Longest token : {longest_token!r}  (len={len(longest_token)})")
    try:
        print(f"Longest token (decoded): {longest_token.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("Longest token: (not valid UTF-8)")

    # Compare with TinyStories tokenizer
    ts_vocab_path = pathlib.Path("output/tinystories_bpe/vocab.json")
    ts_merges_path = pathlib.Path("output/tinystories_bpe/merges.txt")
    if ts_vocab_path.exists() and ts_merges_path.exists():
        ts_vocab = load_bpe_vocab(ts_vocab_path)
        ts_merges = load_bpe_merges(ts_merges_path)
        ts_longest = max(ts_vocab.values(), key=len)

        print(f"\n=== Comparison: TinyStories vs OWT ===")
        print(f"{'':30s} {'TinyStories':>15} {'OWT':>15}")
        print(f"{'Vocab size':30s} {len(ts_vocab):>15} {len(vocab):>15}")
        print(f"{'Merges count':30s} {len(ts_merges):>15} {len(merges):>15}")
        print(f"{'Longest token len':30s} {len(ts_longest):>15} {len(longest_token):>15}")

        ts_token_set = set(ts_vocab.values())
        owt_token_set = set(vocab.values())
        overlap = ts_token_set & owt_token_set
        print(f"{'Token overlap':30s} {len(overlap):>15}")

        ts_long = sorted([t.decode('utf-8', errors='replace') for t in ts_vocab.values() if len(t) > 4], key=len, reverse=True)[:5]
        owt_long = sorted([t.decode('utf-8', errors='replace') for t in vocab.values() if len(t) > 4], key=len, reverse=True)[:5]
        print(f"\nTop-5 longest TinyStories tokens: {ts_long}")
        print(f"Top-5 longest OWT tokens:         {owt_long}")
    else:
        print("\n(TinyStories vocab not found, skipping comparison. Run train_bpe_tinystories.py first.)")
