# Implements SPEC-0001 SS 3.1 (mock task schema).
"""Pydantic models for the mock task corpus.

The corpus lives at `tests/mock_tasks/*.jsonl`. Each line is a serialised
`MockTask`. See SPEC-0001 SS 3.1 for the field contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(StrEnum):
    """Four task types from proposal 05 SS 1."""

    KIS = "KIS"
    QA = "QA"
    AD_HOC = "AD_HOC"
    TRAKE = "TRAKE"


class GroundTruth(BaseModel):
    """Ground-truth payload for one mock task.

    Exactly one of the four optional fields must be populated, matching the
    `task_type` of the parent task. Validation lives on `MockTask`.
    """

    model_config = ConfigDict(extra="forbid")

    kis_frame_ids: list[str] | None = Field(
        default=None,
        description="KIS: list of acceptable frame_ids; submitting any one counts as correct.",
    )
    qa_answer: str | None = Field(
        default=None,
        description="QA: canonical expected text answer.",
    )
    qa_answer_acceptable: list[str] = Field(
        default_factory=list,
        description="QA: alternative phrasings of the canonical answer.",
    )
    adhoc_frame_ids: list[str] | None = Field(
        default=None,
        description="Ad-hoc: pool of relevant frame_ids; partial credit per the scoring formula.",
    )
    trake_frame_ids: list[str] | None = Field(
        default=None,
        description="TRAKE: ordered list of exactly 4 frame_ids.",
    )


class MockTask(BaseModel):
    """A single mock task in our internal evaluation corpus."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        description="Stable identifier, e.g. 'KIS-0001'. Used for seeding deterministic behaviour.",
    )
    task_type: TaskType
    query_vi: str = Field(description="The Vietnamese query as the operator would see it.")
    query_en: str | None = Field(
        default=None,
        description="Optional English paraphrase for cross-lingual debugging.",
    )
    time_limit_seconds: int = Field(
        gt=0,
        description="Per-task wall-clock budget; matches DRES.",
    )
    ground_truth: GroundTruth
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance, difficulty tags, place labels, etc.",
    )

    @model_validator(mode="after")
    def _ground_truth_matches_task_type(self) -> MockTask:
        gt = self.ground_truth
        match self.task_type:
            case TaskType.KIS:
                if not gt.kis_frame_ids:
                    raise ValueError("task_type=KIS requires ground_truth.kis_frame_ids")
            case TaskType.QA:
                if not gt.qa_answer:
                    raise ValueError("task_type=QA requires ground_truth.qa_answer")
            case TaskType.AD_HOC:
                if not gt.adhoc_frame_ids:
                    raise ValueError("task_type=AD_HOC requires ground_truth.adhoc_frame_ids")
            case TaskType.TRAKE:
                if not gt.trake_frame_ids:
                    raise ValueError("task_type=TRAKE requires ground_truth.trake_frame_ids")
                if len(gt.trake_frame_ids) != 4:
                    raise ValueError(
                        "task_type=TRAKE requires exactly 4 ground_truth.trake_frame_ids; "
                        f"got {len(gt.trake_frame_ids)}"
                    )
        return self
