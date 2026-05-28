# Mock task corpus

This directory holds the **internal evaluation corpus** consumed by
`bin/eval` (see [SPEC-0001](../../docs/specs/SPEC-0001-evaluation-harness.md)).

## Files

| File | Tasks | Purpose |
|---|---|---|
| `smoke_20.jsonl` | 20 | Per-PR CI smoke. Loaded by `tests/integration/test_eval_smoke_AC1.py` and by manual `eval` runs during dev. |
| `full_300.jsonl` | -- | Not yet authored. Tracked separately as a workstream per [SPEC-0001 SS 9 Q1](../../docs/specs/SPEC-0001-evaluation-harness.md). |

## Schema

Each line is a JSON object conforming to `aic2026.models.MockTask`. See
[`src/aic2026/models/task.py`](../../src/aic2026/models/task.py) and
[SPEC-0001 SS 3.1](../../docs/specs/SPEC-0001-evaluation-harness.md).

The smoke distribution:

| Task type | Count | Time limit |
|---|---|---|
| KIS | 8 | 300 s |
| QA | 6 | 180 s |
| AD_HOC | 3 | 180 s |
| TRAKE | 3 | 180 s |

Difficulty mix: roughly one-third easy / one-third medium / one-third hard
per type. Place labels span studio, school, outdoor_street, market.

## Important: these are PLACEHOLDERS

The 20 tasks in `smoke_20.jsonl` are **synthetic placeholders** authored
2026-05-28 to exercise the harness mechanics. They reference invented
`vid_*` and `f_*` frame identifiers that do **not** correspond to any
real video corpus. Their purpose is solely to:

1. Make the schema validator have something to load.
2. Give the stub backend deterministic ground truth to hit-or-miss.
3. Let metrics aggregation produce non-trivial numbers.

The **real corpus** (`full_300.jsonl`, 300 tasks) is a separate workstream
described in SPEC-0001 SS 9 Q1: 50 translated from the LSC archive, 50
from AIC HCMC public materials, 100 team-generated, 100 adversarial via
Gemini. Authoring it is ~1 engineer-week and is gated by access to the
2026 dataset (Phase 1, post-June-25).

Vietnamese text uses ASCII transliteration (no diacritics) for now to
sidestep the same Windows-PowerShell encoding gotchas documented in
[AGENTS.md](../../AGENTS.md). The real corpus will use proper diacritics
once Phase 1 ingestion is ready and the encoding pipeline is validated.

## Adding a task

1. Pick a unique `task_id` matching the pattern `<TYPE>-NNNN`.
2. Append one JSON-line to the appropriate file.
3. Validate locally:
   ```
   uv run python -c "from aic2026.models import MockTask; import json; \
     [MockTask.model_validate_json(l) for l in open('tests/mock_tasks/smoke_20.jsonl')]"
   ```
4. Re-run `bin/eval` to confirm the smoke set still produces a valid
   `metrics.json`.
