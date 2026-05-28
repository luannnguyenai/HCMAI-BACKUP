# Proves SPEC-0001 AC4: metrics.json must round-trip through json.load and
# carry all the expected keys.

from __future__ import annotations

import json
from pathlib import Path

from aic2026.harness.backend import StubBackend
from aic2026.harness.runner import EvalRunner, RunConfig, load_tasks
from aic2026.models.metrics import AggregateMetrics, TaskMetrics
from aic2026.models.task import TaskType
from aic2026.reporting.json_writer import write_metrics_json


def _run_smoke(tmp_path: Path) -> AggregateMetrics:
    tasks = load_tasks(Path("tests/mock_tasks/smoke_20.jsonl"))
    backend = StubBackend(seed=42, simulate_latency=False)
    config = RunConfig(
        system="test",
        tasks_path=Path("tests/mock_tasks/smoke_20.jsonl"),
        output_dir=tmp_path,
        seed=42,
    )
    return EvalRunner(backend, config).run(tasks)


def test_metrics_json_loadable_AC4(tmp_path: Path) -> None:
    """AC4: metrics.json on disk parses back into the same shape."""
    metrics = _run_smoke(tmp_path)
    out = tmp_path / "metrics.json"
    write_metrics_json(metrics, out)

    with out.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    expected_top_keys = {
        "schema_version",
        "system",
        "run_id",
        "git_sha",
        "n_tasks",
        "by_task_type",
        "overall",
        "latency",
        "tasks",
    }
    assert expected_top_keys.issubset(payload.keys()), payload.keys()

    # Round-trip through the Pydantic model.
    AggregateMetrics.model_validate(payload)


def test_metrics_json_required_metric_keys_present_AC4(tmp_path: Path) -> None:
    """Each TaskTypeAggregate carries the metrics from proposal 05 SS 4."""
    metrics = _run_smoke(tmp_path)
    out = tmp_path / "metrics.json"
    write_metrics_json(metrics, out)

    with out.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    for task_type in [t.value for t in TaskType]:
        bucket = payload["by_task_type"][task_type]
        for key in (
            "n",
            "n_correct",
            "mean_r_at_1",
            "mean_r_at_5",
            "mean_r_at_10",
            "mean_mrr",
            "mean_kis_score",
            "mean_adhoc_score",
            "wrong_submissions_per_task",
        ):
            assert key in bucket, f"missing {key!r} in {task_type}"

    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "n"):
        assert key in payload["latency"], payload["latency"]


def test_per_task_rows_present_AC4(tmp_path: Path) -> None:
    """`tasks` array carries one TaskMetrics row per loaded task."""
    metrics = _run_smoke(tmp_path)
    assert len(metrics.tasks) == 20
    for row in metrics.tasks:
        TaskMetrics.model_validate(row.model_dump())
