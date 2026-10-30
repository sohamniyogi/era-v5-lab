# Roofline estimator

`perfkit/roofline.py` estimates forward-pass cost from the configured model shapes. It reports FLOPs,
estimated bytes, and a rough kernel count for each timed forward region, then converts them into compute,
memory-bandwidth, and launch lower bounds. The predicted region time is the maximum of those three bounds.

The estimates deliberately simplify real execution. Byte counts assume limited cache reuse, kernel counts are
heuristic, attention assumes a flash-style linear-size IO path, and the default launch latency is 5 microseconds.
Change `--launch-us` or `--bandwidth-gbs` when a measured device profile is available. These estimates help
form profiling hypotheses; use measured timings to judge performance.

From `teams/06-performance`, estimate and compare against an existing measured run:

```bash
python -m perfkit.roofline --config configs/lab_small.yaml \
  --gpu "NVIDIA A100-SXM4-40GB" --run results/runs/baseline_v0.json
```

The command prints an aligned table in the same plain-text style as the correctness summary. It shows the
estimated FLOPs and bytes, compute/memory/launch ceilings, predicted bound, and—when `--run` is supplied—the
measured median and measured-to-estimated ratio for matching regions. A large ratio can point to launch gaps,
synchronization, memory traffic beyond the simple model, or inefficient kernels. A ratio near one is not a
correctness target: the predictions are lower-bound estimates.

Rows marked `total` are the three top-level rows added into the total. `sum` rows roll up child regions and
`part` rows are individual estimates; do not add a sum row to its parts.

To validate across devices, save a `--modes clean,layer` run JSON for each GPU and run the estimator with that
GPU name and matching config, batch size, and sequence length. The committed measured layer traces in this
checkout are A100 runs only; validation on H100/H200/B200/consumer GPUs requires those server run files.

## Model

For each region the estimator reports:

```text
compute_ms   = FLOPs / (dense peak TFLOP/s * 1e9)
bandwidth_ms = estimated bytes / (memory bandwidth GB/s * 1e6)
launch_ms    = estimated kernels * launch latency in microseconds / 1000
region_ms    = max(compute_ms, bandwidth_ms, launch_ms)
```

Inclusive rows such as `fwd:block0` and `fwd:stack` roll up their children and are not added again to the total.
The total covers `fwd:embed`, `fwd:stack`, and `fwd:head+loss`; backward, optimizer, data loading, and distributed
communication are excluded. Dense peaks come from `perfkit/metrics.py`; the bandwidth table in `roofline.py`
uses theoretical device specifications, not measured sustained bandwidth.
