# Implements SPEC-0001 SS 4 (main run loop), AC1, AC2, AC4.
"""End-to-end evaluation runner.

The runner is the orchestration layer: load tasks -> dispatch to backend ->
score -> aggregate -> hand the result back to the CLI for persistence. It
intentionally knows nothing about `metrics.json` or HTML; those concerns
live in `aic2026.reporting`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from aic2026.harness.aggregator import aggregate
from aic2026.harness.backend import Backend
from aic2026.harness.scoring import score_task
from aic2026.models.metrics import AggregateMetrics, TaskMetrics
from aic2026.models.submission import FailureKind
from aic2026.models.task import MockTask

logger = logging.getLogger(__name__)


class TaskLoadError(ValueError):
    """Raised when the mock-task corpus fails schema validation.

    The exception message includes the file path, line number, and the
    underlying Pydantic error so users see exactly where the corpus is bad.
    Drives SPEC-0001 AC2.
    """


@dataclass
class RunConfig:
    """Minimal config carried through one `bin/eval` invocation."""

    system: str
    tasks_path: Path
    output_dir: Path
    seed: int = 42
    concurrency: int = 1  # Tier 1 only supports 1; Tier 2 will lift this.


# --- Task loading ----------------------------------------------------------


def load_tasks(path: Path) -> list[MockTask]:
    """Load mock tasks from a `.jsonl` file or directory of files.

    On schema failure raises `TaskLoadError` with file:line provenance.
    """
    if not path.exists():
        raise TaskLoadError(f"tasks path does not exist: {path}")

    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise TaskLoadError(f"no *.jsonl files found in {path}")
    else:
        files = [path]

    tasks: list[MockTask] = []
    for file_path in files:
        with file_path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise TaskLoadError(f"{file_path}:{line_no}: invalid JSON: {exc.msg}") from exc
                try:
                    tasks.append(MockTask.model_validate(payload))
                except ValidationError as exc:
                    # Compact the Pydantic error tree to the first complaint
                    # so the user sees an actionable message immediately.
                    first_err = exc.errors()[0]
                    loc = ".".join(str(part) for part in first_err.get("loc", ()))
                    msg = first_err.get("msg", "validation failed")
                    raise TaskLoadError(f"{file_path}:{line_no}: {loc}: {msg}") from exc
    if not tasks:
        raise TaskLoadError(f"no tasks found in {path}")
    # Deterministic ordering per SPEC-0001 SS 4.3.
    tasks.sort(key=lambda t: t.task_id)
    return tasks


# --- Run loop --------------------------------------------------------------


class EvalRunner:
    """Single-threaded runner. Tier 2 will add an async/concurrent variant."""

    def __init__(self, backend: Backend, config: RunConfig) -> None:
        self.backend = backend
        self.config = config

    def run(self, tasks: list[MockTask]) -> AggregateMetrics:
        per_task: list[TaskMetrics] = []
        for task in tasks:
            metrics = self._run_one(task)
            per_task.append(metrics)

        by_type, overall, latency = aggregate(per_task)
        return AggregateMetrics(
            schema_version="1",
            system=self.config.system,
            run_id=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            git_sha=_safe_git_sha(),
            n_tasks=len(per_task),
            by_task_type=by_type,
            overall=overall,
            latency=latency,
            tasks=per_task,
        )

    # ---

    def _run_one(self, task: MockTask) -> TaskMetrics:
        budget_ms = task.time_limit_seconds * 1000
        start = time.perf_counter()
        failure_kind: FailureKind | None = None
        try:
            submissions = self.backend.search(task, time_budget_ms=budget_ms)
        except Exception as exc:
            logger.warning("backend failure on %s: %s", task.task_id, exc)
            failure_kind = FailureKind.BACKEND_DOWN
            submissions = []
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        scored = score_task(task, submissions, elapsed_ms)
        return TaskMetrics(
            task_id=task.task_id,
            task_type=task.task_type,
            ok=bool(scored["ok"]) if failure_kind is None else False,
            failure_kind=failure_kind,
            r_at_1=float(scored["r_at_1"]),
            r_at_5=float(scored["r_at_5"]),
            r_at_10=float(scored["r_at_10"]),
            mrr=float(scored["mrr"]),
            time_to_first_correct_ms=scored["time_to_first_correct_ms"],  # type: ignore[arg-type]
            kis_score=scored["kis_score"],  # type: ignore[arg-type]
            adhoc_score=scored["adhoc_score"],  # type: ignore[arg-type]
            adhoc_correct=scored["adhoc_correct"],  # type: ignore[arg-type]
            adhoc_incorrect=scored["adhoc_incorrect"],  # type: ignore[arg-type]
            wrong_submissions=int(scored["wrong_submissions"]),
            end_to_end_ms=float(scored["end_to_end_ms"]),
        )


# --- Provenance helpers ----------------------------------------------------


def _safe_git_sha() -> str | None:
    """Try to read the current git SHA without crashing if git is unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
