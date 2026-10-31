#!/usr/bin/env python3
"""
check_ids.py
============

Gate for v2 (#163): a new tokenizer keeps every id a model may have been trained
with. Compares <new> with <old> (v1) and fails unless

  - every special token (added_tokens) has the same id, content and flags,
  - every id below --base-end maps to the same token string,
  - the first base merges are identical and in the same order,
  - the vocab size is the same.

Ids at or above --base-end (extension and script blocks) may differ; the script
reports how many tokens there kept their id.

Usage
-----
    python check_ids.py /path/to/v1/tokenizer.json /path/to/v2/tokenizer.json --base-end 114333
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--base-end", type=int, default=114333, help="first id after the base block (v1: 114333)")
    args = ap.parse_args()

    old, new = (json.load(open(p, encoding="utf-8")) for p in (args.old, args.new))
    problems = []
    if len(old["model"]["vocab"]) + len(old["added_tokens"]) != len(new["model"]["vocab"]) + len(new["added_tokens"]):
        problems.append("vocab size differs")
    if old["added_tokens"] != new["added_tokens"]:
        problems.append("added_tokens (special ids) differ")

    o_id = {i: t for t, i in old["model"]["vocab"].items()}
    n_id = {i: t for t, i in new["model"]["vocab"].items()}
    base_ids = [i for i in o_id if i < args.base_end]
    diff = [i for i in base_ids if n_id.get(i) != o_id[i]]
    if diff:
        problems.append(f"{len(diff)} base ids differ, first {diff[:5]}")

    base_tokens = {o_id[i] for i in base_ids}
    n_base_merges = sum(1 for a, b in old["model"]["merges"] if a + b in base_tokens)
    if old["model"]["merges"][:n_base_merges] != new["model"]["merges"][:n_base_merges]:
        problems.append(f"the first {n_base_merges} (base) merges differ")

    above = [i for i in o_id if i >= args.base_end]
    same_above = sum(1 for i in above if n_id.get(i) == o_id[i])
    print(f"specials: {len(old['added_tokens'])} identical" if old["added_tokens"] == new["added_tokens"] else "specials: DIFFER")
    print(f"base ids 0..{args.base_end - 1}: {len(base_ids) - len(diff)}/{len(base_ids)} identical; "
          f"base merges: {n_base_merges}")
    print(f"ids >= {args.base_end}: {same_above}/{len(above)} keep their v1 token (expected to change)")
    for p in problems:
        print("FAIL:", p)
    print("RESULT:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
