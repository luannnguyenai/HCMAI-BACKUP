# Proves SPEC-0001 AC2: schema validator rejects malformed task entries
# with a useful, file:line-aware error.

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from aic2026.harness.runner import TaskLoadError, load_tasks


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_rejects_missing_task_type_with_file_and_line_AC2(tmp_path: Path) -> None:
    """AC2: missing required `task_type` field produces a useful error."""
    bad = _write(
        tmp_path / "bad.jsonl",
        [
            json.dumps(
                {
                    "task_id": "BAD-0001",
                    # task_type omitted on purpose
                    "query_vi": "Hello",
                    "time_limit_seconds": 180,
                    "ground_truth": {"qa_answer": "hi"},
                }
            )
        ],
    )

    with pytest.raises(TaskLoadError) as exc_info:
        load_tasks(bad)

    msg = str(exc_info.value)
    assert "bad.jsonl" in msg
    assert ":1:" in msg, f"expected file:line in message, got {msg!r}"
    assert "task_type" in msg


def test_rejects_invalid_json_with_line_AC2(tmp_path: Path) -> None:
    """Bonus AC2 coverage: malformed JSON also surfaces the line number."""
    bad = _write(
        tmp_path / "bad-json.jsonl",
        [
            dedent("{'task_id': 'BAD',  # not valid JSON, single quotes }").strip(),
        ],
    )

    with pytest.raises(TaskLoadError) as exc_info:
        load_tasks(bad)

    assert ":1:" in str(exc_info.value)


def test_rejects_wrong_ground_truth_for_task_type_AC2(tmp_path: Path) -> None:
    """Cross-field validator: task_type=KIS without kis_frame_ids is rejected."""
    bad = _write(
        tmp_path / "mismatch.jsonl",
        [
            json.dumps(
                {
                    "task_id": "KIS-BAD",
                    "task_type": "KIS",
                    "query_vi": "x",
                    "time_limit_seconds": 300,
                    # Wrong: qa_answer for a KIS task
                    "ground_truth": {"qa_answer": "x"},
                }
            )
        ],
    )

    with pytest.raises(TaskLoadError) as exc_info:
        load_tasks(bad)

    assert "kis_frame_ids" in str(exc_info.value)


def test_loads_smoke_corpus_cleanly() -> None:
    """The committed smoke corpus must always parse."""
    smoke = Path("tests/mock_tasks/smoke_20.jsonl")
    tasks = load_tasks(smoke)
    assert len(tasks) == 20
    # Deterministic ordering (SPEC-0001 SS 4.3): sorted by task_id.
    assert tasks == sorted(tasks, key=lambda t: t.task_id)
