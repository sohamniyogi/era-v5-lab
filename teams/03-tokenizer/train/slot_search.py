#!/usr/bin/env python3
"""
slot_search.py
==============

Chooses the v2 id split (#163, v2_spec.md steps 2-3) on held-out text only:
FineWeb-2 `test` for the 11 Indic languages, FineWeb / codeparrot for English
and code (fetch_heldout.py, dev sets). FLORES and the code benchmarks are not read.

1. Extension. o200k_ext_v2.json ranks extension tokens for code and for
   English separately (select_o200k_ext.py). Build v1 base + the first c code
   tokens and measure dev code (tokens / char), and base + the first e English
   tokens and measure dev English (tokens / word), each against
   BrahmicTokenizer-131K on the same text. Take the smallest c and e within
   --target, check the combination, and round its size K up to a multiple of
   --step with more English tokens. Script merges do not change English or code,
   so they are left out here. The chosen `tokens` are written into the json.
2. Script curves. Each script is trained once, on base + K, for up to
   slots_v1 + --extra merges. BPE is greedy and each script trains on its own
   words, so the first n merges equal a run with n slots: one run gives dev
   fertility for every n (step --step).
3. Allocation. 16,384 - K ids go to the scripts. Each script first gets enough
   slots that none of its languages is worse than BrahmicTokenizer-131K on dev
   (+ --lang-floor; at least --floor slots), then --step at a time to the script
   whose next merges lower the dev mean Indic fertility most. Without the
   per-language floor the mean favours small scripts and Hindi ends up worse
   than the baseline.

Usage
-----
    python slot_search.py --corpus /path/to/corpus --heldout /path/to/heldout \\
        --brahmic /path/to/BrahmicTokenizer-131K/tokenizer.json --work /path/to/work

Writes slots_v2.json next to this script, and <work>/slot_search.json (all curves).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_bpe as tb  # noqa: E402
from select_o200k_ext import V1_BASE, combine, with_tokens  # noqa: E402

INDIC = ["Hi", "Bn", "Ta", "Te", "Kn", "Ml", "Mr", "Gu", "Pa", "Or", "As"]


def measure(tok, path, per="word"):
    """Tokens per whitespace word (per="word") or per character (per="char") of a held-out file."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    chunks = ["\n".join(lines[i:i + 2000]) for i in range(0, len(lines), 2000)]
    n = sum(len(e.ids) for e in tok.encode_batch(chunks, add_special_tokens=False))
    return n / (len(text.split()) if per == "word" else len(text))


class Builder:
    """Tokenizers made of the v1 base, the first K extension tokens and some script merges."""

    def __init__(self, norm, pattern):
        from tokenizers import Tokenizer
        self.Tokenizer = Tokenizer
        self.norm_json = tb.normalizer_json(norm) if norm else None
        self.pattern = pattern
        self.base, self.base_merges, _ = tb.build_base(V1_BASE)

    def core(self, ext_hex):
        """(id -> bytes, merges as byte pairs) for base + the given extension tokens (hex)."""
        vocab = dict(self.base)
        merges = self.base_merges + tb.build_ext(vocab, [bytes.fromhex(h) for h in ext_hex], len(vocab))
        return {i: b for b, i in vocab.items()}, merges

    def tokenizer(self, id_bytes, merges, normalize=True):
        vocab = {tb.to_unicode(b): i for i, b in id_bytes.items()}
        j = tb.tokenizer_json(vocab, [(tb.to_unicode(a), tb.to_unicode(b)) for a, b in merges], [], self.pattern,
                              self.norm_json if normalize else None)
        return self.Tokenizer.from_str(json.dumps(j))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, help="training corpus (fetch_corpus.py), as for v1")
    ap.add_argument("--heldout", required=True, help="fetch_heldout.py output; only dev/ is read")
    ap.add_argument("--brahmic", required=True, help="BrahmicTokenizer-131K tokenizer.json")
    ap.add_argument("--work", required=True, help="directory for slot_search.json (outside the repo)")
    ap.add_argument("--template", default=str(HERE / "slots_v1.json"))
    ap.add_argument("--ext", default=str(HERE / "o200k_ext_v2.json"))
    ap.add_argument("--target", type=float, default=0.01, help="max dev En / code excess over the baseline")
    ap.add_argument("--code-step", type=int, default=256)
    ap.add_argument("--en-step", type=int, default=512)
    ap.add_argument("--extra", type=int, default=1536, help="merges past slots_v1 to train per script")
    ap.add_argument("--step", type=int, default=128)
    ap.add_argument("--floor", type=int, default=512)
    ap.add_argument("--lang-floor", type=float, default=0.0,
                    help="each language at most this much worse than BrahmicTokenizer-131K on dev (0 = not worse)")
    args = ap.parse_args()

    import tiktoken
    from tokenizers import Tokenizer
    t_start = time.time()
    dev = Path(args.heldout) / "dev"
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tpl = json.loads(Path(args.template).read_text())
    ext = json.loads(Path(args.ext).read_text())
    ranked = ext["ranked"]
    norm = tb.load_normalizer("v1")
    b = Builder(norm, tiktoken.get_encoding("o200k_base")._pat_str)
    budget = sum(s["slots"] for s in tpl["scripts"].values())

    # 1. Extension size.
    brahmic = Tokenizer.from_file(args.brahmic)
    ref = {"En": measure(brahmic, dev / "En.txt"), "Code": measure(brahmic, dev / "Code.txt", "char")}
    ref.update({lang: measure(brahmic, dev / f"{lang}.txt") for lang in INDIC})
    def rel(name, ext_hex):
        tok = b.tokenizer(*b.core(ext_hex))
        v = measure(tok, dev / f"{name}.txt", "char" if name == "Code" else "word")
        return {"value": v, "vs_ref": v / ref[name] - 1}

    max_k = budget - args.floor * len(tpl["scripts"])
    curves_ext, chosen = {}, {}
    for name, step in (("Code", args.code_step), ("En", args.en_step)):
        curves_ext[name] = []
        for n in range(0, min(len(ranked[name]), max_k) + 1, step):
            p = {"n": n, **rel(name, combine(ranked, n, 0) if name == "Code" else combine(ranked, 0, n))}
            curves_ext[name].append(p)
            print(f"{name} n={n:5d}  {p['value']:.4f} ({p['vs_ref']:+.2%})", flush=True)
            if p["vs_ref"] <= args.target:
                break
        chosen[name] = curves_ext[name][-1]["n"]
        if curves_ext[name][-1]["vs_ref"] > args.target:
            print(f"WARNING: {name} does not reach {args.target:+.1%} within {chosen[name]} tokens", flush=True)
    n_code, n_en = chosen["Code"], chosen["En"]
    while True:                                  # the combination must pass too
        while len(combine(ranked, n_code, n_en)) % args.step:
            n_en += 1
        both = {name: rel(name, combine(ranked, n_code, n_en)) for name in ("En", "Code")}
        if all(v["vs_ref"] <= args.target for v in both.values()) or n_en >= len(ranked["En"]):
            break
        n_en += args.step
    ext = with_tokens(ext, n_code, n_en)
    k = len(ext["tokens"])
    Path(args.ext).write_text(json.dumps(ext, indent=1, ensure_ascii=False) + "\n")
    print(f"chosen: {n_code} code + {n_en} English -> K={k}  En {both['En']['vs_ref']:+.2%}  "
          f"Code {both['Code']['vs_ref']:+.2%}", flush=True)

    # 2. Script curves on base + K.
    id_bytes0, merges0 = b.core(ext["tokens"])
    seg = b.tokenizer(id_bytes0, merges0, normalize=False)    # words are normalized by word_counts already
    curves = {}
    for script, spec in tpl["scripts"].items():
        t0 = time.time()
        counts = tb.word_counts([Path(args.corpus) / f"{l}.txt" for l in spec["langs"]], script, b.pattern, norm)
        items = sorted(counts.items())
        words = [e.ids for e in seg.encode_batch([w.decode("utf-8") for w, _ in items], add_special_tokens=False)]
        id_bytes, vocab_bytes = dict(id_bytes0), set(id_bytes0.values())
        n_max = spec["slots"] + args.extra
        learned = tb.learn_merges(words, [c for _, c in items], id_bytes, vocab_bytes, script, n_max, len(id_bytes0))
        curves[script] = {}
        for n in range(args.floor, n_max + 1, args.step):
            ids = dict(id_bytes0)
            ids.update({nid: id_bytes[nid] for _, _, nid, _ in learned[:n]})
            tok = b.tokenizer(ids, merges0 + [(id_bytes[a], id_bytes[c]) for a, c, _, _ in learned[:n]])
            curves[script][n] = {l: measure(tok, dev / f"{l}.txt") for l in spec["langs"]}
        print(f"{script}: {len(items)} words, curve {args.floor}..{n_max}: "
              + "  ".join(f"{l} {curves[script][args.floor][l]:.4f}->{curves[script][n_max - (n_max - args.floor) % args.step][l]:.4f}"
                          for l in spec["langs"]) + f"  ({time.time() - t0:.0f}s)", flush=True)

    # 3. Allocation.
    def mean_indic(alloc):
        f = {l: v for s, n in alloc.items() for l, v in curves[s][n].items()}
        return sum(f[l] for l in INDIC) / len(INDIC), f

    alloc = {}
    for s, c in curves.items():
        ok = [n for n in sorted(c) if all(c[n][l] <= ref[l] * (1 + args.lang_floor) for l in c[n])]
        if not ok:
            sys.exit(f"ERROR: {s} cannot reach the per-language floor within {max(c)} slots; raise --extra.")
        alloc[s] = ok[0]
    print(f"per-language floor ({args.lang_floor:+.0%} vs BrahmicTokenizer-131K): {alloc}", flush=True)
    left = budget - k - sum(alloc.values())
    if left < 0:
        sys.exit(f"ERROR: the per-language floor needs {sum(alloc.values())} script slots, only {budget - k} left.")
    while left > 0:
        best = None
        for s in alloc:
            n = alloc[s] + args.step
            if n not in curves[s]:
                continue
            gain = sum(curves[s][alloc[s]][l] - curves[s][n][l] for l in curves[s][n]) / len(INDIC)
            if best is None or gain > best[0]:
                best = (gain, s)
        if best is None:
            sys.exit("ERROR: curves too short for the budget; raise --extra.")
        alloc[best[1]] += args.step
        left -= args.step
    v1_alloc = {s: spec["slots"] for s, spec in tpl["scripts"].items()}
    v1_mean, v1_f = mean_indic(v1_alloc)
    v2_mean, v2_f = mean_indic(alloc)
    print(f"dev mean Indic: v1 split on base+K {v1_mean:.4f}  ->  v2 split {v2_mean:.4f} ({v2_mean / v1_mean - 1:+.2%})")
    for s in alloc:
        print(f"  {s:<11} {v1_alloc[s]:5d} -> {alloc[s]:5d}")

    slots = {
        "_comment": "v2 (#163): v1 base kept (ids 0-114332), then an o200k extension block (o200k_ext_v2.json, first "
                    "`slots` tokens), then one block per script. Extension size and script slots chosen by "
                    "slot_search.py on held-out FineWeb-2 / FineWeb / codeparrot dev text, not FLORES. "
                    "Use with --special-layout v0 --normalizer v1.",
        "vocab_size": tpl["vocab_size"], "special_slots": tpl["special_slots"],
        "named_specials": tpl["named_specials"], "base": V1_BASE,
        "o200k_ext": {"slots": k, "tokens": Path(args.ext).name},
        "scripts": {s: {"slots": alloc[s], "langs": spec["langs"]} for s, spec in tpl["scripts"].items()},
    }
    (HERE / "slots_v2.json").write_text(json.dumps(slots, indent=2, ensure_ascii=False) + "\n")
    (work / "slot_search.json").write_text(json.dumps({
        "reference_brahmic_dev": ref, "ext_curves": curves_ext, "ext_chosen": ext["chosen"], "ext_combined_dev": both, "k": k, "script_curves": curves,
        "v1_alloc": v1_alloc, "v2_alloc": alloc, "dev_mean_indic": {"v1_split": v1_mean, "v2_split": v2_mean},
        "dev_per_lang": {"v1_split": v1_f, "v2_split": v2_f}, "seconds": round(time.time() - t_start)}, indent=1))
    print(f"wrote {HERE / 'slots_v2.json'} and {work / 'slot_search.json'}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
