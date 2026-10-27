# G6 -> G8: MFU method and system metrics (week 1)

Hi G8, this is G6's handoff for the training-efficiency numbers.

- Full write-up: `teams/06-performance/docs/METRICS.md`
- The formulas live only in `perfkit/metrics.py`
- Example run: `teams/06-performance/results/runs/baseline_v0.json` on branch `g6/perf-parts-system`
  (A100-40GB, commit `f1f4986`). It will be in `g6/dev` and PR #74 shortly.

## 1. What to read

Every bench run writes one JSON file. The fields you can rely on are listed in `perfkit/schema.py` `REQUIRED`:

| Section | Fields |
|---|---|
| `metrics` | `tokens_per_step`, `tokens_per_s`, `model_flops_per_token`, `hw_flops_per_token`, `params_total`, `params_active`, `peak_tflops`, `model_tflops_per_s`, `mfu`, `hfu` |
| `clean.step_ms` | median, p10, p90, IQR and n of the step time |
| `system` | GPU samples taken during the measured steps (CUDA only) |
| `meta` | GPU, commit, `dirty` flag, torch/CUDA/driver versions, config and time |

Please use the stored values and don't recompute them. If you need a field renamed or removed, ask us first,
because the report and other readers depend on these names. Adding new fields is fine.

## 2. MFU method

```text
model_flops_per_token = 6 * N_active + 12 * L * d * T
hw_flops_per_token    = model + 2 extra stack forwards (reversible: invert + recompute)
                        + 2 * head (only when the loss is chunked)
                        + Muon Newton-Schulz FLOPs per step / tokens per step
MFU = tokens_per_s * model_flops_per_token / (peak_tflops * 1e12)
HFU = same with hw_flops_per_token
```

`T` = sequence length, `L` = layers, `d` = width.

Conventions:

- PaLM / torchtitan style. Attention scores are counted non-causal.
- `N_active` counts only matmul weights a token passes through: attention, the router, the shared expert
  plus the top-2 experts, the KE projection and the LM head. It does not count lookup tables, norms or biases.
- The peak is the dense vendor peak for the run's dtype, with no sparsity, taken from the `PEAKS` table
  in `metrics.py`.
- `tokens_per_s` = tokens per step / median clean step time. The step includes loading data.
- HFU also counts recompute and optimizer FLOPs, so it shows what the hardware actually computed.

Worked example (baseline_v0):

| | |
|---|---:|
| N_active | 51,300,352 |
| 6 * N_active | 307.8M |
| Attention term (12 * L * d * T) | 14.2M |
| model_flops_per_token | 321,957,888 |
| tokens_per_s | 32,340 |
| model TFLOP/s | 10.41 |
| Peak (A100 bf16 dense) | 312 TFLOPS |
| MFU | 3.34% |
| HFU | 4.25% |

Unknown GPU or dtype: `peak_tflops`, `mfu` and `hfu` are `null`. Show them as empty, never as 0.

## 3. System metrics (`run["system"]`)

These come from `nvidia-smi --loop-ms=1000` running during the measured steps (77 samples in the baseline).
Each metric has min, median, max and mean.

| Key | Meaning |
|---|---|
| `gpu_util_pct` | % of the sample period in which at least one kernel ran |
| `mem_util_pct` | % of the sample period in which device memory was read or written |
| `mem_used_mib` | device memory in use, all processes |
| `sm_clock_mhz`, `mem_clock_mhz` | clocks |
| `power_w`, `temp_c` | board power and GPU temperature |
| `slowdown_samples`, `slowdown_reasons` | samples with a power cap, thermal or hardware slowdown, and which ones |

Warning: `gpu_util_pct` only means "at least one kernel was running". It does not mean the SMs were busy.
The baseline shows 59% from nvidia-smi but only 24.9% GPU-busy in the trace (`trace.gpu_busy_pct`).
Plot `gpu_busy_pct` for real utilization.

## 4. Rules for G8 dashboards

- Flag runs where `meta.dirty = true`, or leave them out of baselines.
- Flag runs with any `slowdown_samples`.
- Always show the GPU name and commit next to a number.

## 5. Not done yet

Goodput, HBM GB/s for the roofline, and multi-GPU runs. For now the numbers are for one GPU.

## Questions for G8

- Which format do you ingest: the JSON as it is, or CSV/Parquet?
- Do you need any field we haven't listed?