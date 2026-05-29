# Proves SPEC-0020 AC2: NDCG@10 is emitted to metrics.json on every per-task
# row and every aggregate bucket, the file round-trips, and schema_version is "2".

from __future__ import annotations

import json
from pathlib import Path

from aic2026.harness.backend import StubBackend
from aic2026.harness.runner import EvalRunner, RunConfig, load_tasks
from aic2026.models.metrics import AggregateMetrics
from aic2026.models.task import TaskType
from aic2026.reporting.json_writer import write_metrics_json

_SMOKE = Path("tests/mock_tasks/smoke_20.jsonl")


def _run_smoke(tmp_path: Path) -> AggregateMetrics:
    tasks = load_tasks(_SMOKE)
    backend = StubBackend(seed=42, simulate_latency=False)
    config = RunConfig(system="test", tasks_path=_SMOKE, output_dir=tmp_path, seed=42)
    return EvalRunner(backend, config).run(tasks)


def test_ndcg_present_in_metrics_json_AC2(tmp_path: Path) -> None:
    metrics = _run_smoke(tmp_path)
    out = tmp_path / "metrics.json"
    write_metrics_json(metrics, out)

    with out.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    # schema bumped to v2 for the additive ndcg fields.
    assert payload["schema_version"] == "2"

    # Aggregate roll-ups carry mean_ndcg_at_10.
    assert "mean_ndcg_at_10" in payload["overall"]
    for task_type in [t.value for t in TaskType]:
        assert "mean_ndcg_at_10" in payload["by_task_type"][task_type]

    # Every per-task row carries ndcg_at_10 in [0, 1].
    assert payload["tasks"], "expected at least one task row"
    for row in payload["tasks"]:
        assert "ndcg_at_10" in row
        assert 0.0 <= row["ndcg_at_10"] <= 1.0

    # Round-trips through the strict (extra='forbid') model.
    AggregateMetrics.model_validate(payload)
