"""Append-only JSONL event output for live benchmark consumers."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class JsonlWriter:
    """Write one flushed JSON object per line so readers can tail the file during a run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self._lock = threading.Lock()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc, tb):
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
        return False

    def write(self, record: dict) -> None:
        """Append and flush a record. A lock allows the sampler and step loop to share the writer."""
        line = json.dumps(record, separators=(",", ":"), allow_nan=False)
        with self._lock:
            if self._file is None:
                raise RuntimeError("JsonlWriter must be used inside a with block")
            self._file.write(line + "\n")
            self._file.flush()


def gpu_sample_record(*, run_id: str, node_id: str, gpu: dict, metrics: dict,
                      slowdown_flags: list[str] | None, step: int | None,
                      timestamp: str) -> dict:
    """Build the versioned G8 live GPU-sample record; memory remains in MiB for G8 conversion."""
    return {
        "schema_version": 1,
        "event": "gpu_sample",
        "timestamp": timestamp,
        "run_id": run_id,
        "node_id": node_id,
        "gpu": gpu,
        "step": step,
        "metrics": metrics,
        "slowdown_flags": slowdown_flags,
    }
