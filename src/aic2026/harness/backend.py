# Implements SPEC-0001 SS 2.1 (stub backend) and SS 4 (backend protocol).
"""Backend interface plus a deterministic stub used for Tier 1.

The stub serves as a stand-in for a real retrieval system so we can verify
the harness end-to-end without depending on Milvus / Elasticsearch / a VLM.
Future tiers wire the real backend through the same `Backend` protocol.

Determinism trace: each task is processed with `hashlib.sha256(task_id +
str(seed))`; the resulting digest seeds an isolated `random.Random`
instance. The same `(task_id, seed)` always produces the same submissions
and the same simulated latency.
"""

from __future__ import annotations

import hashlib
import random
import time
from typing import Protocol

from aic2026.models.submission import Submission
from aic2026.models.task import GroundTruth, MockTask, TaskType

# Stub behaviour constants. Surfaced here so AGENTS.md "no magic numbers"
# is satisfied and so tests can reference the same values.
TOP_K: int = 10
GROUND_TRUTH_INCLUSION_PROBABILITY: float = 0.7
LATENCY_MIN_MS: float = 10.0
LATENCY_MAX_MS: float = 30.0


class Backend(Protocol):
    """Minimal contract a retrieval backend must satisfy.

    Real backends (Milvus + reranker + VLM stack) implement this in later
    specs. Tier 1 only ships the stub below.
    """

    def search(
        self,
        task: MockTask,
        time_budget_ms: int,
    ) -> list[Submission]:
        """Return up to TOP_K submissions for `task`, ranked best-first."""
        ...


def _seed_rng(task_id: str, seed: int) -> random.Random:
    """Deterministic, hash-based RNG seeding per `(task_id, seed)`."""
    digest = hashlib.sha256(f"{task_id}|{seed}".encode()).digest()
    int_seed = int.from_bytes(digest[:8], "big")
    return random.Random(int_seed)


def _fake_frame_id(rng: random.Random, salt: int) -> str:
    """Construct a plausible-looking but deterministic fake frame id."""
    vid = rng.randint(0, 999)
    frame = rng.randint(0, 9_999)
    # `_salt_{i}` keeps fake ids unique within a single task even if the RNG
    # happens to repeat a vid/frame pair.
    return f"vid_{vid:03d}_f{frame:07d}_salt{salt}"


def _ground_truth_pool(gt: GroundTruth, task_type: TaskType) -> list[str]:
    match task_type:
        case TaskType.KIS:
            return list(gt.kis_frame_ids or [])
        case TaskType.AD_HOC:
            return list(gt.adhoc_frame_ids or [])
        case TaskType.TRAKE:
            return list(gt.trake_frame_ids or [])
        case _:
            return []


class StubBackend:
    """Deterministic stub. Returns:

    - **KIS**: top-10 frame ids; with `GROUND_TRUTH_INCLUSION_PROBABILITY`
      a correct id is inserted at rank `(hash(task_id) mod 10) + 1`. Score
      decays linearly with rank.
    - **AD_HOC**: top-10 frame ids drawing roughly half from the relevant
      pool when included.
    - **TRAKE**: with `GROUND_TRUTH_INCLUSION_PROBABILITY` returns the 4
      correct ground-truth ids in the *correct order*, then 6 fakes; else
      shuffled / wrong fakes only.
    - **QA**: a single `Submission` whose `.text` is the canonical answer
      with `GROUND_TRUTH_INCLUSION_PROBABILITY`, else a placeholder
      "khong biet" ("don't know") string.

    Latency is simulated via `time.sleep` to a uniform value in
    `[LATENCY_MIN_MS, LATENCY_MAX_MS]` drawn from the per-task RNG.
    """

    def __init__(self, seed: int = 0, *, simulate_latency: bool = True) -> None:
        self.seed = seed
        self._simulate_latency = simulate_latency

    def search(self, task: MockTask, time_budget_ms: int) -> list[Submission]:
        rng = _seed_rng(task.task_id, self.seed)
        latency_ms = rng.uniform(LATENCY_MIN_MS, LATENCY_MAX_MS)
        if self._simulate_latency:
            # Cap simulated sleep at the task's budget to avoid blowing
            # past time limits during heavy CI parallelism.
            sleep_ms = min(latency_ms, max(time_budget_ms - 1, 0))
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        include_truth = rng.random() < GROUND_TRUTH_INCLUSION_PROBABILITY

        if task.task_type is TaskType.QA:
            return self._qa_submissions(task, rng, include_truth)
        if task.task_type is TaskType.TRAKE:
            return self._trake_submissions(task, rng, include_truth)
        return self._frame_submissions(task, rng, include_truth)

    # --- per-task-type ------------------------------------------------------

    def _frame_submissions(
        self,
        task: MockTask,
        rng: random.Random,
        include_truth: bool,
    ) -> list[Submission]:
        pool = _ground_truth_pool(task.ground_truth, task.task_type)
        # Insertion rank chosen deterministically from the task_id hash so
        # different tasks land at different ranks across the corpus.
        digest_int = int.from_bytes(
            hashlib.sha256(task.task_id.encode("utf-8")).digest()[:4], "big"
        )
        insert_rank = (digest_int % TOP_K) + 1  # 1-based, 1..TOP_K

        submissions: list[Submission] = []
        for rank in range(1, TOP_K + 1):
            score = max(0.0, 1.0 - (rank - 1) * 0.07)
            if include_truth and pool and rank == insert_rank:
                # Pick a relevant id deterministically from the pool.
                fid = pool[rng.randrange(len(pool))]
            else:
                fid = _fake_frame_id(rng, salt=rank)
            submissions.append(Submission(rank=rank, score=score, frame_id=fid))

        if task.task_type is TaskType.AD_HOC and include_truth and pool:
            # Sprinkle a few more correct ids in to make Ad-hoc recall
            # non-trivial. Replace ranks 6..7 with relevant ids when
            # available.
            for extra_rank in (6, 7):
                if rng.random() < 0.5 and pool:
                    fid = pool[rng.randrange(len(pool))]
                    submissions[extra_rank - 1] = Submission(
                        rank=extra_rank,
                        score=submissions[extra_rank - 1].score,
                        frame_id=fid,
                    )

        return submissions

    def _trake_submissions(
        self,
        task: MockTask,
        rng: random.Random,
        include_truth: bool,
    ) -> list[Submission]:
        pool = _ground_truth_pool(task.ground_truth, task.task_type)
        submissions: list[Submission] = []
        if include_truth and len(pool) == 4:
            for rank, fid in enumerate(pool, start=1):
                submissions.append(
                    Submission(rank=rank, score=1.0 - (rank - 1) * 0.05, frame_id=fid),
                )
            # Pad to TOP_K with fakes.
            for rank in range(5, TOP_K + 1):
                submissions.append(
                    Submission(
                        rank=rank,
                        score=max(0.0, 1.0 - (rank - 1) * 0.07),
                        frame_id=_fake_frame_id(rng, salt=rank),
                    ),
                )
        else:
            # Shuffle the ground truth so order is wrong (or all-fake).
            scrambled = list(pool)
            rng.shuffle(scrambled)
            for rank in range(1, TOP_K + 1):
                fid = (
                    scrambled[rank - 1]
                    if rank - 1 < len(scrambled) and rng.random() < 0.3
                    else _fake_frame_id(rng, salt=rank)
                )
                submissions.append(
                    Submission(
                        rank=rank,
                        score=max(0.0, 1.0 - (rank - 1) * 0.07),
                        frame_id=fid,
                    ),
                )
        return submissions

    def _qa_submissions(
        self,
        task: MockTask,
        rng: random.Random,
        include_truth: bool,
    ) -> list[Submission]:
        gt = task.ground_truth
        text = gt.qa_answer if include_truth and gt.qa_answer else "khong biet"
        _ = rng  # rng unused beyond the upstream `include_truth` draw
        return [Submission(rank=1, score=1.0, text=text)]
