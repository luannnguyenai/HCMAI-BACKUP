# Implements SPEC-0001 SS 3.3 (metrics.json shape) and AC4.
# Implements SPEC-0020 SS 3 (NDCG@10 fields + schema_version bump).
"""Aggregate and per-task metric shapes emitted by `bin/eval`.

The `AggregateMetrics` instance is what gets serialised to `metrics.json` at
the end of a run. The CI gate (SPEC-0001 AC7, Tier 3) reads this file.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aic2026.models.submission import FailureKind
from aic2026.models.task import TaskType


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    mean_ms: float = Field(ge=0.0)
    n: int = Field(ge=0)


class TaskMetrics(BaseModel):
    """One row per scored task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_type: TaskType
    ok: bool
    failure_kind: FailureKind | None = None

    # Retrieval correctness; for QA these collapse to {0, 1}.
    r_at_1: float = Field(ge=0.0, le=1.0)
    r_at_5: float = Field(ge=0.0, le=1.0)
    r_at_10: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    # NDCG@10 (SPEC-0020); binary-gain. C2 learned-fusion ship-gate metric.
    ndcg_at_10: float = Field(ge=0.0, le=1.0)

    # KIS-specific (null on other task types).
    time_to_first_correct_ms: float | None = Field(default=None, ge=0.0)
    kis_score: float | None = Field(default=None, ge=0.0, le=100.0)

    # Ad-hoc-specific (null on other task types).
    adhoc_score: float | None = Field(default=None, ge=0.0, le=100.0)
    adhoc_correct: int | None = Field(default=None, ge=0)
    adhoc_incorrect: int | None = Field(default=None, ge=0)

    # Common.
    wrong_submissions: int = Field(ge=0)
    end_to_end_ms: float = Field(ge=0.0)

    # Calibration surface, populated by future tiers.
    brier_score: float | None = None


class TaskTypeAggregate(BaseModel):
    """Per-task-type roll-up. Used both for individual task types and the overall slice."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=0)
    n_correct: int = Field(ge=0)
    mean_r_at_1: float = Field(ge=0.0, le=1.0)
    mean_r_at_5: float = Field(ge=0.0, le=1.0)
    mean_r_at_10: float = Field(ge=0.0, le=1.0)
    mean_mrr: float = Field(ge=0.0, le=1.0)
    mean_ndcg_at_10: float = Field(ge=0.0, le=1.0)  # SPEC-0020
    mean_kis_score: float | None = None
    mean_adhoc_score: float | None = None
    wrong_submissions_per_task: float = Field(ge=0.0)


class AggregateMetrics(BaseModel):
    """Top-level shape persisted to `metrics.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="2",
        description=(
            "Bump on breaking changes; consumers may pin a major. "
            "v2 (SPEC-0020): added ndcg_at_10 / mean_ndcg_at_10."
        ),
    )
    system: str
    run_id: str
    git_sha: str | None = None
    n_tasks: int = Field(ge=0)
    by_task_type: dict[TaskType, TaskTypeAggregate]
    overall: TaskTypeAggregate
    latency: LatencyStats
    tasks: list[TaskMetrics] = Field(
        default_factory=list,
        description="Per-task detail rows. Useful for slicing in `report.html`.",
    )
