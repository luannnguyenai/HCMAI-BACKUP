# Deterministic behaviour of the stub backend.

from __future__ import annotations

import pytest

from aic2026.harness.backend import StubBackend
from aic2026.models.task import GroundTruth, MockTask, TaskType


def _task(task_id: str, task_type: TaskType) -> MockTask:
    if task_type is TaskType.KIS:
        gt = GroundTruth(kis_frame_ids=[f"correct_{task_id}_1", f"correct_{task_id}_2"])
        time_limit = 300
    elif task_type is TaskType.AD_HOC:
        gt = GroundTruth(adhoc_frame_ids=[f"rel_{task_id}_{i}" for i in range(4)])
        time_limit = 180
    elif task_type is TaskType.TRAKE:
        gt = GroundTruth(trake_frame_ids=[f"trake_{task_id}_{i}" for i in range(4)])
        time_limit = 180
    elif task_type is TaskType.QA:
        gt = GroundTruth(qa_answer="forty-two")
        time_limit = 180
    else:  # pragma: no cover
        raise AssertionError(task_type)
    return MockTask(
        task_id=task_id,
        task_type=task_type,
        query_vi="placeholder",
        time_limit_seconds=time_limit,
        ground_truth=gt,
    )


def _backend() -> StubBackend:
    return StubBackend(seed=123, simulate_latency=False)


@pytest.mark.parametrize("task_type", [TaskType.KIS, TaskType.AD_HOC])
def test_returns_top_10_submissions(task_type: TaskType) -> None:
    out = _backend().search(_task("T-001", task_type), time_budget_ms=10_000)
    assert len(out) == 10
    assert [s.rank for s in out] == list(range(1, 11))
    # Scores monotonically non-increasing (allowing equality).
    scores = [s.score for s in out]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_qa_returns_single_text_submission() -> None:
    out = _backend().search(_task("Q-001", TaskType.QA), time_budget_ms=10_000)
    assert len(out) == 1
    assert out[0].text is not None
    assert out[0].frame_id is None


def test_trake_can_return_4_correct_frames() -> None:
    """At seed=123 + sufficient retries we eventually hit the include-truth branch.

    We don't assert any specific task's outcome; only that across many tasks
    the stub is non-trivial (i.e. produces SOME exact-match TRAKE sequences).
    """
    backend = StubBackend(seed=0, simulate_latency=False)
    hits = 0
    for i in range(50):
        task = _task(f"TR-{i:03d}", TaskType.TRAKE)
        out = backend.search(task, time_budget_ms=10_000)
        if task.ground_truth.trake_frame_ids is None:
            continue
        if [s.frame_id for s in out[:4]] == task.ground_truth.trake_frame_ids:
            hits += 1
    # 50 tries at 70% include-truth probability has Pr(<= 10 hits) ~ 0.
    assert hits >= 20, f"expected at least 20 exact-match TRAKE hits, got {hits}"


def test_deterministic_given_task_and_seed() -> None:
    """Same (task_id, seed) -> identical submission stream."""
    task = _task("KIS-001", TaskType.KIS)
    b1 = StubBackend(seed=7, simulate_latency=False)
    b2 = StubBackend(seed=7, simulate_latency=False)
    out1 = b1.search(task, time_budget_ms=10_000)
    out2 = b2.search(task, time_budget_ms=10_000)
    assert [s.frame_id for s in out1] == [s.frame_id for s in out2]
    assert [s.score for s in out1] == [s.score for s in out2]


def test_different_seeds_produce_different_outputs() -> None:
    task = _task("KIS-001", TaskType.KIS)
    out_a = StubBackend(seed=1, simulate_latency=False).search(task, time_budget_ms=10_000)
    out_b = StubBackend(seed=2, simulate_latency=False).search(task, time_budget_ms=10_000)
    # Stream contents diverge across seeds (insertion logic + fakes both seed-dependent).
    assert [s.frame_id for s in out_a] != [s.frame_id for s in out_b]
