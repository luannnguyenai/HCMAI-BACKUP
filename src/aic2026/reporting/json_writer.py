# Implements SPEC-0001 SS 3.3 (metrics.json) and AC4.
"""Serialise `AggregateMetrics` to JSON on disk."""

from __future__ import annotations

import json
from pathlib import Path

from aic2026.models.metrics import AggregateMetrics


def write_metrics_json(metrics: AggregateMetrics, path: Path) -> None:
    """Pretty-print `metrics` to `path` as UTF-8 JSON.

    Pydantic v2 `.model_dump(mode="json")` handles enums + datetime
    serialisation, which is exactly what AC4 requires (`json.load` must
    round-trip the result).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
