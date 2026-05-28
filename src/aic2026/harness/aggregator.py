# Implements SPEC-0001 SS 3.3 and SS 4 (aggregation, per-task-type slicing).
"""Aggregation utilities. Pure: no I/O, deterministic given inputs."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean

from aic2026.models.metrics import LatencyStats, TaskMetrics, TaskTypeAggregate
from aic2026.models.task import TaskType


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def compute_latency_stats(values_ms: Iterable[float]) -> LatencyStats:
    xs = sorted(float(v) for v in values_ms)
    if not xs:
        return LatencyStats(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, mean_ms=0.0, n=0)
    return LatencyStats(
        p50_ms=_percentile(xs, 0.50),
        p95_ms=_percentile(xs, 0.95),
        p99_ms=_percentile(xs, 0.99),
        mean_ms=mean(xs),
        n=len(xs),
    )


def aggregate_by_task_type(tasks: list[TaskMetrics]) -> TaskTypeAggregate:
    """Roll up a (possibly per-task-type) list into a single aggregate."""
    if not tasks:
        return TaskTypeAggregate(
            n=0,
            n_correct=0,
            mean_r_at_1=0.0,
            mean_r_at_5=0.0,
            mean_r_at_10=0.0,
            mean_mrr=0.0,
            mean_kis_score=None,
            mean_adhoc_score=None,
            wrong_submissions_per_task=0.0,
        )
    n_correct = sum(1 for t in tasks if t.ok)
    kis_scores = [t.kis_score for t in tasks if t.kis_score is not None]
    adhoc_scores = [t.adhoc_score for t in tasks if t.adhoc_score is not None]
    return TaskTypeAggregate(
        n=len(tasks),
        n_correct=n_correct,
        mean_r_at_1=mean(t.r_at_1 for t in tasks),
        mean_r_at_5=mean(t.r_at_5 for t in tasks),
        mean_r_at_10=mean(t.r_at_10 for t in tasks),
        mean_mrr=mean(t.mrr for t in tasks),
        mean_kis_score=mean(kis_scores) if kis_scores else None,
        mean_adhoc_score=mean(adhoc_scores) if adhoc_scores else None,
        wrong_submissions_per_task=mean(t.wrong_submissions for t in tasks),
    )


def aggregate(
    tasks: list[TaskMetrics],
) -> tuple[
    dict[TaskType, TaskTypeAggregate],
    TaskTypeAggregate,
    LatencyStats,
]:
    """Build (by_task_type, overall, latency) from per-task metrics."""
    by_type: dict[TaskType, TaskTypeAggregate] = {}
    for task_type in TaskType:
        bucket = [t for t in tasks if t.task_type is task_type]
        by_type[task_type] = aggregate_by_task_type(bucket)
    overall = aggregate_by_task_type(tasks)
    latency = compute_latency_stats(t.end_to_end_ms for t in tasks)
    return by_type, overall, latency
