"""Formula tests for the FLOP and hardware-utilization metrics sent with benchmark runs."""
import pytest
import torch

from models.config import ModelConfig
from models.gpt import GPT
from perfkit.metrics import flops_per_token, mfu, peak_tflops


def _tiny_model(**overrides):
    cfg = ModelConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=8, **overrides)
    return GPT(cfg)


def test_flops_per_token_matches_documented_model_and_hardware_formulas():
    model = _tiny_model()
    seq_len = 5
    _, active = model.param_counts()
    head = model.head.weight.numel()
    model_fpt = 6 * active + 12 * model.cfg.n_layer * model.cfg.n_embd * seq_len
    stack_fwd = 2 * (active - head) + 4 * model.cfg.n_layer * model.cfg.n_embd * seq_len

    result = flops_per_token(model, seq_len, opt_flops_per_step=600, tokens_per_step=10)

    assert result["model_flops_per_token"] == model_fpt
    assert result["hw_flops_per_token"] == pytest.approx(model_fpt + 60)
    assert result["params_total"] == sum(p.numel() for p in model.parameters())
    assert result["params_active"] == active
    assert stack_fwd > 0  # documents the reversible/chunked extra-work basis below


@pytest.mark.parametrize(
    ("overrides", "extra"),
    [
        ({"reversible": True}, "reversible"),
        ({"ce_impl": "chunked"}, "chunked"),
    ],
)
def test_flops_per_token_counts_extra_reversible_or_chunked_work(overrides, extra):
    model = _tiny_model(**overrides)
    seq_len = 5
    _, active = model.param_counts()
    head = model.head.weight.numel()
    ke = model.tok.proj.weight.numel() if model.cfg.embedding == "ke1" else 0
    model_fpt = 6 * active + 12 * model.cfg.n_layer * model.cfg.n_embd * seq_len
    expected = model_fpt
    if extra == "reversible":
        stack_fwd = 2 * (active - head - ke) + 4 * model.cfg.n_layer * model.cfg.n_embd * seq_len
        expected += 2 * stack_fwd
    else:
        expected += 2 * head
    result = flops_per_token(model, seq_len)
    assert result["model_flops_per_token"] == model_fpt
    assert result["hw_flops_per_token"] == expected


def test_mfu_and_hfu_use_model_and_hardware_flops_respectively():
    result = mfu({"model_flops_per_token": 2e9, "hw_flops_per_token": 3e9}, 1000, 100)
    assert result["model_tflops_per_s"] == pytest.approx(2.0)
    assert result["mfu"] == pytest.approx(0.02)
    assert result["hfu"] == pytest.approx(0.03)


def test_mfu_without_peak_leaves_utilization_unknown_but_reports_throughput():
    result = mfu({"model_flops_per_token": 2e9}, 1000, None)
    assert result == {"model_tflops_per_s": pytest.approx(2.0), "mfu": None, "hfu": None}


@pytest.mark.parametrize(
    ("gpu", "dtype", "expected"),
    [("NVIDIA A100-SXM4-80GB", "bf16", (312, "A100")),
     ("NVIDIA H100 NVL", "bf16", (None, "H100 NVL")),
     ("unlisted accelerator", "bf16", (None, None)),
     ("NVIDIA V100", "bf16", (None, "V100"))],
)
def test_peak_tflops_known_unknown_gpu_and_dtype(gpu, dtype, expected):
    assert peak_tflops(gpu, dtype) == expected
