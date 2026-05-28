# Implements SPEC-0001 SS 3 (backend submission shapes).
"""Shapes describing what a backend returns for one task and how the harness
records the result of each scored attempt.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FailureKind(StrEnum):
    """Stable failure taxonomy logged alongside each task result."""

    BACKEND_DOWN = "backend_down"
    SCHEMA_INVALID = "schema_invalid"
    WALL_CLOCK_TIMEOUT = "wall_clock_timeout"
    DRES_UNREACHABLE = "dres_unreachable"
    OTHER = "other"


class Submission(BaseModel):
    """One candidate item returned by a backend for a given task.

    For KIS / Ad-hoc / TRAKE this is a `frame_id`. For QA this is a `text`
    field carrying the candidate answer. Exactly one of the two is populated.
    """

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, description="1-based rank within the returned list.")
    score: float = Field(description="Backend's own score; ordering matches rank.")
    frame_id: str | None = Field(
        default=None,
        description="KIS / Ad-hoc / TRAKE candidate frame identifier.",
    )
    text: str | None = Field(
        default=None,
        description="QA candidate answer text.",
    )


class SubmissionResult(BaseModel):
    """Outcome of one (task, attempted_submission) pair.

    The harness persists one row per submission to `submissions.parquet` in
    Tier 3; Tier 1 uses the in-memory list to drive scoring + aggregation.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    rank: int
    correct: bool
    latency_ms: float
    failure_kind: FailureKind | None = None
