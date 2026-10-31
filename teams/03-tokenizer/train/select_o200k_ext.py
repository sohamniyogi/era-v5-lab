#!/usr/bin/env python3
"""
select_o200k_ext.py
===================

Ranks the o200k_base tokens past the v1 base cut by how many tokens they save
on held-out English and on held-out code, as two separate lists (#163,
v2_spec.md step 1). The base itself is not touched: its ids stay as in v1.

For each admissible candidate t (<= 32 bytes, at most one script, rank past the
cut) and each domain d in {En, Code}, with counts from encoding the usage set
with full o200k:

    gain_d(t) = count_d(t) * (pieces(t) - 1)

pieces(t) is the number of base tokens t splits into. Counts are robust to
bursts: the usage text is cut into --chunk-kb pieces, a token counts at most
--cap times per piece, and it must occur in --min-chunks pieces of that domain
(one generated file must not pick tokens for everyone).

The domains are ranked separately because each has its own limit (at most
+1% vs BrahmicTokenizer-131K): one mixed list kept adding code tokens long
after code met its limit. Each list brings missing o200k merge ancestors first,
so every prefix is closed under merges. slot_search.py picks a prefix of each
list and writes the combined `tokens` that train_bpe.py reads (combine()).

Usage
-----
    python select_o200k_ext.py --heldout /path/to/heldout --out o200k_ext_v2.json
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bpe import MAX_BYTES, build_base, o200k_split, scripts_of  # noqa: E402

V1_BASE = 114332          # tokens in the v1 base (ids 0-114332, id 36 is a special)


def pieces(token, base, ranks):
    """Tokens `token` splits into using only base tokens (lowest-rank merge first, as tiktoken)."""
    parts = [bytes([x]) for x in token]
    while len(parts) > 1:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and parts[i] + parts[i + 1] in base and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
    return len(parts)


def usage_counts(enc, path, chunk_kb, cap):
    """Capped o200k token counts (by rank), chunk counts, and the file size in MB."""
    text = Path(path).read_text(encoding="utf-8")
    counts, chunks = Counter(), Counter()
    step = chunk_kb * 1000
    for i in range(0, len(text), step):
        c = Counter(enc.encode_ordinary(text[i:i + step]))
        counts.update({r: min(n, cap) for r, n in c.items()})
        chunks.update(c.keys())
    return counts, chunks, len(text.encode("utf-8")) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", required=True, help="fetch_heldout.py output (uses usage/En.txt, usage/Code.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=8192, help="length of the ranked list")
    ap.add_argument("--chunk-kb", type=int, default=20)
    ap.add_argument("--cap", type=int, default=4, help="max count of a token per chunk")
    ap.add_argument("--min-chunks", type=int, default=3)
    args = ap.parse_args()

    import tiktoken
    t0 = time.time()
    enc = tiktoken.get_encoding("o200k_base")
    ranks = enc._mergeable_ranks
    by_rank = {r: t for t, r in ranks.items()}
    base, _, _ = build_base(V1_BASE)
    cut = max(ranks[t] for t in base)

    def admissible(t):
        return len(t) <= MAX_BYTES and len(scripts_of(t)) <= 1

    def rank(gain):
        """Candidates in gain order, each after its missing ancestors, up to --max tokens."""
        chosen, have = [], set(base)
        def add(t):
            if t in have:
                return
            for part in o200k_split(ranks, t, ranks[t]):
                add(part)
            have.add(t)
            chosen.append(t)
        for t in sorted(gain, key=lambda t: (-gain[t], ranks[t])):
            mark = len(chosen)
            add(t)
            if len(chosen) > args.max:          # ancestors overflowed the budget: undo this candidate
                del chosen[mark:]
                break
        return chosen

    usage, ranked = {}, {}
    for name in ("En", "Code"):
        counts, chunks, mb = usage_counts(enc, Path(args.heldout) / "usage" / f"{name}.txt", args.chunk_kb, args.cap)
        gain = {by_rank[r]: c * (pieces(by_rank[r], base, ranks) - 1) for r, c in counts.items()
                if r > cut and by_rank[r] not in base and admissible(by_rank[r]) and chunks[r] >= args.min_chunks}
        ranked[name] = rank(gain)
        usage[name] = {"mb": round(mb, 3), "capped_tokens": sum(counts.values()), "candidates": len(gain)}
        print(f"{name}: {len(gain)} candidates; first {[t.decode('utf-8', 'replace') for t in ranked[name][:15]]}",
              flush=True)

    out = {
        "_comment": "o200k_base tokens past the v1 base cut, ranked by tokens saved on held-out English and on "
                    "held-out code (two lists, best first, each prefix closed under o200k merges with the v1 base). "
                    "`tokens` is the combination slot_search.py chose; train_bpe.py reads it. Made by "
                    "select_o200k_ext.py and slot_search.py (#163).",
        "base_tokens": V1_BASE, "base_cut_rank": cut,
        "counting": {"chunk_kb": args.chunk_kb, "cap": args.cap, "min_chunks": args.min_chunks},
        "usage": usage,
        "ranked": {name: [t.hex() for t in ts] for name, ts in ranked.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"cut rank {cut}; wrote {args.out} ({time.time() - t0:.0f}s)")


def combine(ranked, n_code, n_en):
    """The first n_code code tokens, then the first n_en English tokens (hex), without repeats.
    Both prefixes are closed under merges, so the result is too."""
    out, seen = [], set()
    for h in ranked["Code"][:n_code] + ranked["En"][:n_en]:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def with_tokens(data, n_code, n_en):
    """`data` (an o200k_ext json) with the chosen combination as `tokens` and its sha256."""
    tokens = combine(data["ranked"], n_code, n_en)
    return {**data, "chosen": {"Code": n_code, "En": n_en}, "tokens": tokens,
            "tokens_sha256": hashlib.sha256("\n".join(tokens).encode()).hexdigest()}


if __name__ == "__main__":
    main()
