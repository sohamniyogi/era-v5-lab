"""Per-GPU MFU/HFU values and the JSON reason-code contract."""
from perfkit.metrics import mfu, peak_unavailable_reason
from perfkit.schema import example_run, validate_run


def test_mfu_hfu_are_values_with_null_reasons_when_peak_is_known():
    out = mfu({"model_flops_per_token": 2e12, "hw_flops_per_token": 3e12}, 100, 100)

    assert out["mfu"] == 2.0
    assert out["hfu"] == 3.0
    assert out["mfu_unavailable_reason"] is None
    assert out["hfu_unavailable_reason"] is None


def test_missing_peak_sets_separate_reason_fields_for_both_metrics():
    out = mfu({"model_flops_per_token": 2e12, "hw_flops_per_token": 3e12}, 100, None,
              peak_reason="unsupported_dtype")

    assert out["mfu"] is None and out["hfu"] is None
    assert out["mfu_unavailable_reason"] == "unsupported_dtype"
    assert out["hfu_unavailable_reason"] == "unsupported_dtype"


def test_hfu_can_be_unavailable_while_mfu_is_available():
    out = mfu({"model_flops_per_token": 2e12}, 100, 100)

    assert out["mfu"] == 2.0
    assert out["mfu_unavailable_reason"] is None
    assert out["hfu"] is None
    assert out["hfu_unavailable_reason"] == "hw_flops_unavailable"


def test_peak_unavailable_reason_distinguishes_gpu_dtype_and_bad_override():
    assert peak_unavailable_reason("unlisted device", "bf16") == "unknown_gpu"
    assert peak_unavailable_reason("NVIDIA A100", "fp8") == "unsupported_dtype"
    assert peak_unavailable_reason("NVIDIA A100", "bf16", -1) == "invalid_peak_tflops"
    assert peak_unavailable_reason("NVIDIA A100", "bf16", 312) is None


def test_schema_accepts_null_metric_with_reason_and_rejects_unknown_reason():
    run = example_run()
    run["metrics"]["mfu"] = None
    run["metrics"]["mfu_unavailable_reason"] = "unknown_gpu"
    assert validate_run(run) == []

    run["metrics"]["hfu_unavailable_reason"] = "vendor-magic"
    assert any("metrics.hfu_unavailable_reason" in error for error in validate_run(run))
