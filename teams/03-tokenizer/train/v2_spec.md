# v2 spec: o200k extension by usage + slot splits chosen on held-out text (#163)

## Problem

v1-c1 passes the bar, but English is at **+1.98%** against a 2.00% limit (FLORES En vs BrahmicTokenizer-131K). The
base keeps the first 114,332 admissible o200k ranks. o200k's rank order follows OpenAI's training mix, so many
English and code tokens past the cut are missing, while the base contains many CJK, Cyrillic and other tokens that
our model will rarely see. The v1 slot split was hand-picked and checked on FLORES, which is also the eval set.

## Fixed (do not change in v2)

| What | Value | Why |
|---|---|---|
| Vocab size | 131,072 | G4 sizes the embedding from it |
| Special ids | tokenizer v0's 356 entries at their exact ids | #107; never reuse a trained id |
| **Base ids** | **ids 0–114,332 (114,332 tokens, id 36 is a special), byte-identical to v1** | G4/G5 may have trained on them |
| Normalizer | v1 (`nfc+chillu+nukta-BnPa`, #134) | unchanged |
| Pre-tokenizer, byte map, BPE settings | as v1 | unchanged |

Only the 16,384 ids after the base change: they hold the new **o200k extension block** plus the script blocks.

## Data

| Set | Source | Used for | Never used for |
|---|---|---|---|
| Train (Indic) | FineWeb-2 `train`, the v1 corpus (60 MB Ta/Te/Kn/Ml, 20 MB others) | script merges | selection, reporting |
| Dev (Indic) | FineWeb-2 `test`, first 2 MB per language | slot split selection | training |
| Check (Indic) | FineWeb-2 `test`, next 2 MB per language | reporting only (overfit check) | training, selection |
| Usage (En/code) | FineWeb `sample-10BT` 30 MB + `codeparrot-clean-valid` 20 MB | ranking extension tokens | measuring |
| Dev (En/code) | the next 5 MB / 3 MB of the same streams | choosing the extension size | ranking |
| Check (En/code) | the next 5 MB / 3 MB | reporting only | everything else |
| Eval | FLORES-200, HumanEval, MBPP, GSM8K | the COMPARE.md table only | everything else |

- Documents in the train set are removed from every held-out set by exact text hash.
- Code held-out text is checked for 13-gram overlap with HumanEval and MBPP; overlapping files are dropped.
- Held-out files live outside the repo; `fetch_heldout.py` writes a manifest with sizes and sha256.

## Method

### 1. o200k extension (`select_o200k_ext.py`)
- Candidates: o200k tokens with rank above the v1 cut-off (rank 114,483) that are admissible in the same way
  as the base: at most 32 bytes and at most one script. Each one brings its o200k merge parents, so every
  prefix of a ranked list is closed under merges.
- Score per domain d (English, code): `gain_d(t) = count_d(t) × (pieces(t) − 1)`. `count_d` is from encoding the
  usage set with full o200k, and `pieces(t)` is the number of base tokens `t` splits into.
- Counts are robust to bursts: at most 4 per 20 KB chunk, and at least 3 chunks. Without this, one generated
  file put tokens like `\xf` and `_guest` at the top.
- **English and code are ranked separately.** One mixed list kept adding code tokens after code had met its
  limit, and needed K = 6,656 for the same English result.
- `train_bpe.py` reads the combined `tokens` (code prefix, then English prefix), sorts them by o200k rank and
  appends their merges after the base merges. Merge priority therefore stays o200k's own order.

### 2. Extension size and script curves (`slot_search.py`)
- Extension: the smallest code prefix and the smallest English prefix within **+1.0%** of BrahmicTokenizer-131K
  on dev (code tokens/char, English tokens/word). Then a check that the combination passes too, with K rounded
  up to a multiple of 128.
- Scripts: base + extension fixed, each script trained once with up to `slots_v1 + 1,536` merges. BPE is greedy
  and each script trains on its own words, so the first `n` merges equal a run with `n` slots. One run gives
  dev fertility (tokens / whitespace word) for every `n` in steps of 128.

### 3. Allocation
- Budget: `K + Σ slots = 16,384`.
- **Per-language floor:** each script first gets enough slots that none of its languages is worse than
  BrahmicTokenizer-131K on dev (`--lang-floor`, default 0). Without it, the mean favours small scripts:
  Devanagari dropped to 768 slots and Hindi was +9% vs the baseline on FLORES.
- The remaining ids go 128 at a time to the script whose next merges lower the **dev mean Indic fertility**
  the most.

### 4. Build and gates
- `train_bpe.py --slots slots_v2.json --special-layout v0 --normalizer v1`.
- Gates: the 3 verify scripts PASS and the round-trip sample passes (`train_bpe.py` exits 1 otherwise);
  `check_ids.py` PASS: base vocab, base merges and all special ids are byte-identical to v1 (exits 1 otherwise).
- Report: harness `--suites flores,code` plus the check sets. COMPARE.md gets a v2 section.

## Results (full tables in COMPARE.md, section v2)

- K = 4,864 (768 code + 4,096 English). Slots: Devanagari 2,816, Bengali 1,408, Tamil, Telugu, Kannada and
  Malayalam 1,024 each, Gujarati 768, Gurmukhi 896, Oriya 1,536.
- FLORES (report only): English **+0.96%** (v1-c1 +1.98%), HumanEval +0.61%, MBPP +0.81%, mean Indic **−12.49%**
  (v1-c1 −16.97%). All four bars pass.
- Held-out check split: English +0.9%, code +0.8%, mean Indic −12.9%. It agrees with dev and FLORES.
- Gates: 3 verify scripts PASS; `check_ids.py` against v1-c1 PASS; v1-c1 still rebuilds byte-identically.
- Known gap: on FLORES, Bn is +0.2% and Gu +0.4% vs the baseline, although the floor holds on dev.

## Done when

- FLORES En at most **+1.0%** vs BrahmicTokenizer-131K (v1: +1.98%); HumanEval and MBPP at most +2%.
- Mean Indic FLORES still beats BrahmicTokenizer-131K.
- Check-set results move in the same direction as dev (no overfit to the dev split).

## Risks

- Ids for the extension come out of the script blocks, so Indic fertility can get slightly worse. The allocation
  makes that trade explicit; the curves go into COMPARE.md.
- Selection on usage is approximate (counts come from full o200k). The final numbers are always measured with
  the real built tokenizer, never estimated.
- Python is the only code language in the usage set. That matches HumanEval and MBPP, but other languages
  may gain less.

## Next improvements (after v2)

Ordered by expected value. Each becomes its own `[G3]` issue when it starts.

1. **Iterative extension ranking.** Counts now come from full o200k once. Re-encoding with base + the chosen
   tokens after each block of 512, then re-scoring, should reach +1.0% English with fewer ids, because shorter
   tokens that full o200k hides get counted. Every id saved goes back to the Dravidian scripts, which paid the most.
2. **More and cleaner training text.** The scripts train on 20–60 MB per language with no cleaning. Use Data A/B's
   cleaned output, and drop boilerplate (C1) and machine-translated spam (D4) with the diagnosis medicines. In the
   Bengali FineWeb-2 diagnosis pack, 95 of 500 docs had boilerplate and 36 were MT spam, and merges are spent on
   strings like "সাবস্ক্রাইব করুন". Go to 100 MB or more per language, then re-run the slot search.
3. **ZWJ / ZWNJ.** The Bengali FineWeb-2 diagnosis pack has 0 ZWJ/ZWNJ in 1.34M characters (stripped upstream),
   so no merge involving them is learned (for example র‍্যা). Measure their rate in FLORES, IN22 and real user
   text per language, then decide between handling them in the normalizer and putting them back in training text.
4. **Per-language weights in the allocation.** The objective is the plain mean of 11 languages, as in the pass
   bar. Weighting by expected share of training tokens (from Data A/B or G4) would match what the model actually
   sees. Bring this to the team before changing the pass bar.
5. **Wider evaluation.** Add IN22-Gen/Conv and keep the held-out check split as a standing set in the harness.
   Add Indic digits and romanized Indic (Hinglish) probes; romanized text competes with English for base ids.
6. **Odia / Punjabi idea 1 (leading-space tokens).** Re-measure first: v2 is already −40% (Odia) and −22%
   (Punjabi) vs the baseline on the held-out check split.
7. **Test with a model (G4/G5).** Compare v1-c1 and v2 on bits per byte with a small model. A fertility gain is not
   a modelling gain until a model shows it.
