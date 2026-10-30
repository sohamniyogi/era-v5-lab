"""Live JSONL event record contract."""
import json
import io
from types import SimpleNamespace

import pytest
import torch

from perfkit.events import JsonlWriter, gpu_sample_record
from perfkit.metrics import GpuSampler


def test_jsonl_writer_flushes_one_json_record_per_line(tmp_path):
    path = tmp_path / "run.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"event": "first", "step": 1})
        assert path.read_text(encoding="utf-8").splitlines() == ['{"event":"first","step":1}']
        writer.write({"event": "second", "step": 2})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    with pytest.raises(FileExistsError):
        with JsonlWriter(path):
            pass


def test_gpu_sample_record_contains_g8_identity_and_raw_mib_metrics():
    record = gpu_sample_record(
        run_id="run-123", node_id="node-2",
        gpu={"name": "Example GPU", "index": 1, "uuid": "GPU-example", "pci_bus_id": "0000:02:00.0"},
        metrics={"gpu_util_pct": 95, "mem_used_mib": 1200}, slowdown_flags=["SwPowerCap"],
        step=41, timestamp="2026-10-30T12:00:00.000Z",
    )
    assert record == {
        "schema_version": 1, "event": "gpu_sample", "timestamp": "2026-10-30T12:00:00.000Z",
        "run_id": "run-123", "node_id": "node-2",
        "gpu": {"name": "Example GPU", "index": 1, "uuid": "GPU-example", "pci_bus_id": "0000:02:00.0"},
        "step": 41, "metrics": {"gpu_util_pct": 95, "mem_used_mib": 1200},
        "slowdown_flags": ["SwPowerCap"],
    }


def test_gpu_sampler_stream_callback_receives_sample_and_current_step():
    lines = "95, 40, 1200, 1410, 1600, 300.0, 65, 0x0000000000000004\n"
    samples = []
    sampler = GpuSampler(torch.device("cpu"), on_sample=lambda row, step: samples.append((row, step)))
    sampler.reason = "clocks_event_reasons.active"
    sampler._proc = SimpleNamespace(stdout=io.StringIO(lines))
    sampler.set_step(12)

    sampler._read_samples()

    assert len(samples) == 1 and samples[0][1] == 12
    assert samples[0][0]["memory.used"] == 1200
    assert samples[0][0][sampler.reason] == 4
