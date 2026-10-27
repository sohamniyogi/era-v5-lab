# Baseline reproduction: BrahmicTokenizer-131K (tokenizer v0)

Run on 27 Oct 2026, Apple Silicon Mac (10 cores, 16 GB), Python 3.11.5,
tokenizers 0.23.3, transformers 5.21.0, tiktoken 0.14.0, datasets 5.1.0.

Result: **all 4 verify scripts PASS, and FLORES-200 fertility matches the committed reference on all 13 columns with |Δ| = 0.0000** (tolerance 0.001).

## 1. Setup

```bash
git clone https://github.com/theschoolofai/BrahmicTokenizer-131K
cd BrahmicTokenizer-131K            # commit 7dfcf398172c825c6689ad578e8b484ca5f57bc9
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python tokenizers transformers tiktoken pandas pyarrow datasets sentencepiece protobuf
shasum -a 256 tokenizer.json
# 01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff
```

## 2. Verify scripts

zsh does not split `$BRAHMIC` into words in every context, so run the four scripts one by one:

```bash
python verify_no_cross_script_merges.py        ./tokenizer.json
python verify_max_byte_length.py               ./tokenizer.json
python verify_kronecker_constraints_unified.py ./tokenizer.json
python verify_sarvam_m_structural_identity.py  ./tokenizer.json ./tokenizer.json
```

| Script | Exit | Key output |
|---|---|---|
| verify_no_cross_script_merges.py | 0 | merge-rule entries: 301,398 · cross-script: 0 · PASS |
| verify_max_byte_length.py | 0 | vocab 131,072 · normal 130,716 · special 356 · byte-fallback 0 · longest 32 bytes · > 32 bytes: 0 · PASS |
| verify_kronecker_constraints_unified.py | 0 | ByteLevel · max byte length 32 · > 32 bytes: 0 · cross-script tokens: 0 · RESULT: PASS |
| verify_sarvam_m_structural_identity.py | 0 | vocab 131,072 / merges 301,398 (rank order same) / added 356 identical · STRUCTURALLY IDENTICAL |

All four match the expected output in the BrahmicTokenizer-131K README.

## 3. FLORES-200 fertility

```bash
curl -sL -o flores200_dataset.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
shasum -a 256 flores200_dataset.tar.gz
# b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6
tar -xzf flores200_dataset.tar.gz          # -> flores200_dataset/{dev,devtest}

python evaluate_fertility_flores.py --tokenizers BrahmicTokenizer-131K --languages all \
    --flores-dir ./flores200_dataset --out ./brahmic_flores.json
```

2,009 sentences per language (dev 997 + devtest 1,012). Fertility = Σ tokens / Σ whitespace words.

| | En | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean (11 Indic) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reproduced | 1.2352 | 1.6429 | 2.3295 | 3.1166 | 3.0292 | 3.2734 | 3.4738 | 2.5723 | 2.2748 | 2.5955 | 4.1287 | 2.7535 | **2.8355** |
| Reference | 1.2352 | 1.6429 | 2.3295 | 3.1166 | 3.0292 | 3.2734 | 3.4738 | 2.5723 | 2.2748 | 2.5955 | 4.1287 | 2.7535 | 2.8355 |
| \|Δ\| | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Reference = `reference_outputs/evaluate_fertility_flores.json` in the same repo.

## 4. Same check through the team harness

`harness/run_harness.py` re-implements the same method and fails (exit 1) if any column is more than 0.001 away from the reference:

```bash
cd era-v5-lab/teams/03-tokenizer/harness
python run_harness.py --tokenizers BrahmicTokenizer-131K --check-reference \
    --brahmic-dir /path/to/BrahmicTokenizer-131K --flores-dir /path/to/flores200_dataset
# max |d| = 0.0000 -> PASS   (exit 0)
```

The comparator rows (Gemma-4, Qwen, DeepSeek V4, Sarvam, o200k) are in `../benchmarks_v1.md`.
