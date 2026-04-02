"""
train_bpe_tinystories.py

问题 train_bpe_tinystories (a)(b):
  在 TinyStories 上训练字节级 BPE 分词器，vocab_size=10000。
  序列化词表和 merges，打印训练时间、内存峰值、最长 token。

运行方式:
  python train_bpe_tinystories.py

Training BPE on data/TinyStoriesV2-GPT4-train.txt (vocab_size=10000, workers=56) ...

Saved vocab to output/tinystories_bpe/vocab.json
Saved merges to output/tinystories_bpe/merges.txt

=== Results ===
Training time : 456.3s (0.1268 hours)
Peak memory   : 0.16 GB
Vocab size    : 10000
Merges count  : 9743
Longest token : b' accomplishment'  (len=15)
Longest token (decoded): ' accomplishment'

"""
import time
import tracemalloc
import pathlib
import multiprocessing

from tests.adapters import run_train_bpe, save_bpe_vocab, save_bpe_merges

DATA_PATH = pathlib.Path("data/TinyStoriesV2-GPT4-train.txt")
OUT_DIR = pathlib.Path("output/tinystories_bpe")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_WORKERS = multiprocessing.cpu_count()  # 56 on this server

'''
=== Results ===
Training time : 220.2s (0.0612 hours)
Peak memory   : 0.15 GB
Vocab size    : 10000
Merges count  : 9743
Longest token : b' accomplishment'  (len=15)
Longest token (decoded): ' accomplishment'
'''

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
    print(f"\n=== Results ===")
    print(f"Training time : {elapsed:.1f}s ({elapsed/3600:.4f} hours)")
    print(f"Peak memory   : {peak_bytes / 1024**3:.2f} GB")
    print(f"Vocab size    : {len(vocab)}")
    print(f"Merges count  : {len(merges)}")
    print(f"Longest token : {longest_token!r}  (len={len(longest_token)})")
    try:
        print(f"Longest token (decoded): {longest_token.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("Longest token: (not valid UTF-8)")
