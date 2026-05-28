# Proves SPEC-0001 AC1 end-to-end:
#   `eval --tasks tests/mock_tasks/smoke_20.jsonl --system v0.0.1-stub`
# runs to completion and emits report.html + metrics.json.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_eval() -> list[str]:
    """Locate the `eval` entry point for subprocess invocation.

    Order of preference:
      1. `eval` on PATH (uv-installed)
      2. `uv run eval`
      3. `python -m aic2026.cli.eval` as a final fallback
    """
    eval_path = shutil.which("eval")
    if eval_path:
        return [eval_path]
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "run", "eval"]
    return [sys.executable, "-m", "aic2026.cli.eval"]


def test_eval_smoke_end_to_end_AC1(tmp_path: Path) -> None:
    """AC1: the harness produces the three output files and metrics are non-zero."""
    tasks = REPO_ROOT / "tests" / "mock_tasks" / "smoke_20.jsonl"
    assert tasks.exists(), tasks

    cmd = [
        *_find_eval(),
        "--tasks",
        str(tasks),
        "--system",
        "v0.0.1-test",
        "--output",
        str(tmp_path),
        "--no-latency-sim",
    ]

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        f"eval exited {completed.returncode}\nSTDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )

    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.html"
    readme_path = tmp_path / "README.md"

    assert metrics_path.exists(), "metrics.json missing"
    assert report_path.exists(), "report.html missing"
    assert readme_path.exists(), "README.md missing"

    with metrics_path.open(encoding="utf-8") as fh:
        metrics = json.load(fh)

    assert metrics["n_tasks"] == 20
    assert metrics["system"] == "v0.0.1-test"
    overall = metrics["overall"]
    # Stub backend at seed=42 returns the ground truth 70% of the time; mean
    # R@10 across 20 tasks should be comfortably non-zero.
    assert overall["mean_r_at_10"] > 0.0, overall
    assert overall["mean_mrr"] > 0.0, overall

    # report.html exists and contains the system tag for a quick sanity check.
    report_text = report_path.read_text(encoding="utf-8")
    assert "v0.0.1-test" in report_text
    assert "Overall" in report_text


def test_eval_rejects_missing_task_type_AC2(tmp_path: Path) -> None:
    """AC2 end-to-end: malformed corpus exits non-zero with a useful message."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "task_id": "BAD-0001",
                "query_vi": "x",
                "time_limit_seconds": 180,
                "ground_truth": {"qa_answer": "x"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = [
        *_find_eval(),
        "--tasks",
        str(bad),
        "--system",
        "v0.0.1-test",
        "--output",
        str(tmp_path / "out"),
        "--no-latency-sim",
    ]

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )

    assert completed.returncode != 0
    combined = (completed.stdout + completed.stderr).lower()
    assert "task_type" in combined or "validation" in combined, combined
