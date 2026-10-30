# METRICS — MFU / HFU formulas for G8

Source of truth for the numbers: `perfkit/metrics.py`. Bench writes them into the run JSON
(`perfkit/schema.py` `REQUIRED["metrics"]`); report and G8 read the same fields. Do not invent
constants here — change `metrics.py` first.

## FLOPs per token

`T` = sequence length, `L` = layers, `d` = width, `N_active` from `GPT.param_counts`.

```text
model_flops_per_token = 6 * N_active + 12 * L * d * T
                        (PaLM; attention scores counted non-causal)

hw_flops_per_token    = model
                      + 2 forwards of the stack when reversible (invert + recompute)
                      + 2 * head again when ce_impl=chunked (it recomputes the head)
                      + optimizer FLOPs per step (Muon Newton-Schulz) / tokens per step
```

## MFU and HFU

```text
model_tflops_per_s = tokens_per_s * model_flops_per_token / 1e12
MFU                = model_tflops_per_s / peak_tflops

HFU                = tokens_per_s * hw_flops_per_token / 1e12 / peak_tflops
```

MFU uses model FLOPs; HFU uses hardware FLOPs. Both divide by the **dense** vendor peak for the
dtype the run used (`peak_tflops` in the run JSON). The run JSON `metrics` section stores `mfu`, `hfu`,
`model_tflops_per_s`, `model_flops_per_token`, `hw_flops_per_token`, `params_total` and `params_active`;
`mfu` and `hfu` are `null` when there is no peak. The optimizer FLOPs come from `Optimizers.flops()` in
`models/optim.py`.

## Peak table (`PEAKS` in `metrics.py`)

`gpu_peak(device_name)` / `peak_tflops(device_name, dtype)` walk a substring table (B200, H200,
H100, A100, L40S, …). First match wins, so specific names come first.

- Known GPU, known dtype peak → `peak_tflops` is set; MFU/HFU can be computed.
- Known GPU, empty peak entry (today `H100 NVL`, `A10G`), a dtype with no entry (bf16 on V100 / T4 / P100,
  where bench falls back to fp16 anyway), or unknown GPU → peak is missing; **leave MFU empty**. The bench
  adds a run warning; pass `--peak-tflops` on the bench CLI to set it.

## Tokens per second

`tokens_per_s` comes from the clean median step time and `tokens_per_step` (batch × sequence).
Report and G8 must use the value already stored in the run JSON, not recompute a different one.

## Dirty commit

`meta.dirty` is true when the working tree under `teams/06-performance` (outside `results/`)
differs from `meta.commit`, new files included. A dirty run is not a named commit: treat baselines from dirty
trees as non-authoritative until the tree is clean. `perfkit.report.compare` refuses a pair unless both runs
have `dirty: false` and the same commit. `dirty` is `null` when git status cannot run; with no `.git`
at all, `meta.commit` is taken from the `PERF_COMMIT` environment variable (or `null`).

## System metrics (`run["system"]`, CUDA runs)

`perfkit.metrics.GpuSampler` runs `nvidia-smi --loop-ms=1000` in a child process while the clean steps run
(the steps that give step time, tokens/s and MFU), so the samples describe the same steps. Each metric has
`min`, `median`, `max`, `mean` over the samples in the final JSON. The sampler also streams individual records
to the run's `<tag>.jsonl` sidecar as they arrive. Each line is a JSON object with `schema_version`, `event`,
UTC `timestamp`, `run_id`, `node_id`, `gpu` (`name`, local `index`, `uuid`, `pci_bus_id`), current clean `step`
when available, raw `metrics`, and `slowdown_flags`. `mem_used_mib` remains in MiB for G8 to convert. Reading
the JSONL does not synchronize CUDA; a separate thread consumes `nvidia-smi` output and flushes each event.
By default each invocation gets a unique `run_id`; pass `--run-id <id>` to share an ID across coordinated G5/G6
invocations. `--node-id` overrides the default node name (`G8_NODE_ID` or hostname). `--repeat` creates separate
runs and therefore cannot be combined with an explicit shared run ID. The current harness samples the one GPU
selected by that process; multi-GPU/multi-node training collection is not implemented yet.

| Key | nvidia-smi field | Meaning |
|---|---|---|
| `gpu_util_pct` | `utilization.gpu` | % of the sample period in which at least one kernel ran. Not SM occupancy: one tiny kernel at a time reads 100%. Use `trace.gpu_busy_pct` for idle gaps between kernels. |
| `mem_util_pct` | `utilization.memory` | % of the sample period in which device memory was read or written |
| `mem_used_mib` | `memory.used` | device memory in use, all processes (the allocator's own peak is `memory.peak_allocated_gb`) |
| `sm_clock_mhz`, `mem_clock_mhz` | `clocks.sm`, `clocks.mem` | clocks; compare with `meta.telemetry_start["clocks.max.sm"]` |
| `power_w`, `temp_c` | `power.draw`, `temperature.gpu` | board power and GPU temperature |

`slowdown_samples` / `slowdown_reasons`: samples in which the driver reported a power cap, thermal or hardware
slowdown (`SLOWDOWN` in `metrics.py`; idle and application clocks are not slowdowns). Any slowdown sample adds
a run warning: such a run may miss the 5% check. Off CUDA, or without `nvidia-smi`, there is no `system` section.

Besides the sampler, `meta.telemetry_start` and `meta.telemetry` (`after_warmup`, `during_steps`, `end`) hold
single `nvidia-smi` readings (`gpu_telemetry`). The `during_steps` reading is taken while three untimed steps run.
From those, bench warns at 83 C or above, when the driver reports a slowdown, or (when it reports no reasons)
when the SM clock is below 85% of `clocks.max.sm`.

## Goodput (partial)

`run["goodput"]["badput_s"]` has the wall-clock seconds of `startup` (model, loader, optimizer built),
`first_step` (includes `torch.compile` / autotune when on) and `warmup`. Time goodput and token goodput are
not written yet.

## Shared run JSON

`perfkit.schema.REQUIRED` lists every field report and G8 expect under `meta`, `config`, `clean`,
and `metrics` (plus `layer` / `trace` when those modes ran, `system` on CUDA). Renaming or removing a required key
breaks consumers — agree it before changing.
