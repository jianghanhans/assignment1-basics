/*
 * bpe_fast.c — C acceleration for BPE merge loop.
 *
 * Two exported functions:
 *   bpe_count_pairs  — build initial pair counts from CSR pretokens
 *   bpe_full_merge   — apply one merge to all pretokens, return pair deltas
 *
 * Both share an internal open-addressing hash map that aggregates
 * (pair_a, pair_b) -> int64 weighted count deltas.
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC -o tests/bpe_fast.so tests/bpe_fast.c
 */

#include <stdint.h>
#include <string.h>

/* ── Hash map ────────────────────────────────────────────────────────────── */

#define HM_CAP  (1 << 20)           /* 1 048 576 slots — enough for 32K vocab */
#define HM_MASK (HM_CAP - 1)

/* 0x8080808080808080 is what memset(..., 0x80, ...) produces for int64;
   it can never be a valid key (would require token IDs = INT32_MIN/2). */
#define HM_EMPTY ((int64_t)0x8080808080808080LL)

static int64_t _hm_keys[HM_CAP];
static int64_t _hm_vals[HM_CAP];

/* Track insertion order so we can iterate and selectively clear. */
static int32_t _hm_pa[HM_CAP];
static int32_t _hm_pb[HM_CAP];
static int32_t _hm_used;

static int _hm_initialized = 0;

static void _hm_ensure_init(void) {
    if (!_hm_initialized) {
        memset(_hm_keys, 0x80, sizeof(_hm_keys));
        _hm_initialized = 1;
    }
}

/* Selective clear: only reset slots that were written this round. */
static void _hm_clear(void) {
    _hm_ensure_init();
    for (int32_t i = 0; i < _hm_used; i++) {
        int64_t key = ((int64_t)(uint32_t)_hm_pa[i] << 32) | (uint32_t)_hm_pb[i];
        uint32_t h = (uint32_t)(((uint64_t)key * 0x9e3779b97f4a7c15ULL) >> 44) & HM_MASK;
        while (_hm_keys[h] != key)
            h = (h + 1) & HM_MASK;
        _hm_keys[h] = HM_EMPTY;
        /* _hm_vals[h] will be overwritten on next write */
    }
    _hm_used = 0;
}

static inline uint32_t _hm_slot(int32_t pa, int32_t pb) {
    int64_t key = ((int64_t)(uint32_t)pa << 32) | (uint32_t)pb;
    return (uint32_t)(((uint64_t)key * 0x9e3779b97f4a7c15ULL) >> 44) & HM_MASK;
}

static void _hm_add(int32_t pa, int32_t pb, int64_t delta) {
    int64_t key = ((int64_t)(uint32_t)pa << 32) | (uint32_t)pb;
    uint32_t h  = _hm_slot(pa, pb);
    while (_hm_keys[h] != HM_EMPTY && _hm_keys[h] != key)
        h = (h + 1) & HM_MASK;
    if (_hm_keys[h] == HM_EMPTY) {
        _hm_keys[h] = key;
        _hm_vals[h] = delta;
        _hm_pa[_hm_used] = pa;
        _hm_pb[_hm_used] = pb;
        _hm_used++;
    } else {
        _hm_vals[h] += delta;
    }
}

/* Drain non-zero entries from the hash map into caller-provided arrays.
   Returns number of entries written. */
static int32_t _hm_drain(int32_t *out_a, int32_t *out_b, int64_t *out_v,
                          int32_t max_out) {
    int32_t n = 0;
    for (int32_t i = 0; i < _hm_used && n < max_out; i++) {
        int64_t key = ((int64_t)(uint32_t)_hm_pa[i] << 32) | (uint32_t)_hm_pb[i];
        uint32_t h  = _hm_slot(_hm_pa[i], _hm_pb[i]);
        while (_hm_keys[h] != key)
            h = (h + 1) & HM_MASK;
        int64_t v = _hm_vals[h];
        if (v != 0) {
            out_a[n] = _hm_pa[i];
            out_b[n] = _hm_pb[i];
            out_v[n] = v;
            n++;
        }
    }
    return n;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

/*
 * bpe_count_pairs — scan all pretokens and accumulate weighted pair counts.
 *
 *   flat    : concatenated token IDs (int32), modified in-place by merges
 *   starts  : flat[starts[i] .. starts[i]+lens[i]) is pretoken i
 *   lens    : current length of each pretoken
 *   freqs   : occurrence frequency of each pretoken
 *   n       : number of pretokens
 *   out_a,b : output pair IDs
 *   out_cnt : output weighted counts
 *   max_out : capacity of out_* arrays
 *
 * Returns: number of unique pairs written.
 */
int32_t bpe_count_pairs(
    const int32_t *flat,
    const int32_t *starts,
    const int32_t *lens,
    const int32_t *freqs,
    int32_t        n,
    int32_t       *out_a,
    int32_t       *out_b,
    int64_t       *out_cnt,
    int32_t        max_out
) {
    _hm_clear();
    for (int32_t ii = 0; ii < n; ii++) {
        const int32_t *tok  = flat + starts[ii];
        int32_t        tlen = lens[ii];
        int64_t        freq = freqs[ii];
        for (int32_t j = 0; j < tlen - 1; j++)
            _hm_add(tok[j], tok[j + 1], freq);
    }
    return _hm_drain(out_a, out_b, out_cnt, max_out);
}

/*
 * bpe_full_merge — scan all pretokens, apply merge (pa,pb)->new_id in-place,
 *                  accumulate weighted pair deltas.
 *
 * Returns: number of unique pair changes written to out_a/out_b/out_delta.
 */
int32_t bpe_full_merge(
    int32_t       *flat,
    const int32_t *starts,
    int32_t       *lens,
    const int32_t *freqs,
    int32_t        n,
    int32_t        pa,
    int32_t        pb,
    int32_t        new_id,
    int32_t       *out_a,
    int32_t       *out_b,
    int64_t       *out_delta,
    int32_t        max_out
) {
    _hm_clear();

    for (int32_t ii = 0; ii < n; ii++) {
        int32_t *tok  = flat + starts[ii];
        int32_t  tlen = lens[ii];
        int64_t  freq = freqs[ii];

        int32_t wi = 0, ri = 0;
        while (ri < tlen) {
            if (ri + 1 < tlen && tok[ri] == pa && tok[ri + 1] == pb) {
                /* left-neighbor pairs */
                if (wi > 0) {
                    _hm_add(tok[wi - 1], pa,     -freq);
                    _hm_add(tok[wi - 1], new_id, +freq);
                }
                /* right-neighbor pairs */
                if (ri + 2 < tlen) {
                    _hm_add(pb,     tok[ri + 2], -freq);
                    _hm_add(new_id, tok[ri + 2], +freq);
                }
                tok[wi++] = new_id;
                ri += 2;
            } else {
                if (wi != ri) tok[wi] = tok[ri];   /* compact only if shifted */
                wi++; ri++;
            }
        }
        if (wi != tlen) lens[ii] = wi;
    }

    return _hm_drain(out_a, out_b, out_delta, max_out);
}
