---
id: SPEC-0001
title: Evaluation harness (mock-DRES + 300-task set + bin/eval CLI)
status: Draft
owner: unassigned
created: 2026-05-26
updated: 2026-05-26
implements_proposal: docs/proposals/05-evaluation-harness.md
related_adrs:
  - ADR-0008
  - ADR-0009
depends_on: []
---

# SPEC-0001 — Evaluation harness (mock-DRES + 300-task set + bin/eval CLI)

> The internal evaluation harness — a locally-hosted DRES instance, a 300-task mock set, and the `bin/eval` CLI that runs them. This is the gate every other spec runs against. Without it, we are guessing.

## 1. Context

[`docs/proposals/05-evaluation-harness.md`](../proposals/05-evaluation-harness.md) defines the role and design. This spec narrows it to a buildable unit: the harness must be runnable from a single CLI command, must produce comparable numbers across system versions, and must be the CI gate for any PR that affects retrieval / ranking / reranking / fine-tuning quality.

It must also serve as the **substrate for the bakeoff in SPEC-0002** and the ablation gates for the original contributions in ADR-0007.

## 2. Scope

### 2.1 In scope
- Local DRES instance via Docker Compose, seeded with our mock tasks.
- A `tests/mock_tasks/` directory containing 300 mock tasks across KIS / QA / Ad-hoc / TRAKE.
- `bin/eval` CLI: takes a `--tasks` set and `--system` version, runs them, submits to the local DRES, writes a report.
- Per-PR smoke set (20 fixed tasks); nightly full set (300 tasks).
- Per-class slicing (by place / ADL / time-of-day) and per-task-type slicing.
- Metric set: R@1 / R@5 / R@10, MRR, time-to-first-correct, wrong-submission rate, p50/p95 latency.

### 2.2 Out of scope
- Generation of new mock tasks (handled in a separate workflow described below; not gated by this spec).
- The ablation logic for specific contributions (C1/C2/C4 ablations are *consumers* of this harness, not part of it).
- A web dashboard for results (Prometheus + DuckDB only; HTML report is generated per-run).

## 3. API contract

### 3.1 Mock task schema (`tests/mock_tasks/*.jsonl`, one JSON per line)

```python
class MockTask(BaseModel):
    task_id: str                # e.g. "KIS-0001"
    task_type: Literal["KIS", "QA", "AD_HOC", "TRAKE"]
    query_vi: str               # the Vietnamese query as the operator would see it
    query_en: str | None        # optional English paraphrase for cross-lingual debugging
    time_limit_seconds: int     # 300 for KIS, 180 for QA / Ad-hoc / TRAKE
    ground_truth: GroundTruth
    metadata: dict[str, Any]    # provenance, difficulty tags, place labels, etc.

class GroundTruth(BaseModel):
    # KIS: one of frame_ids must be submitted
    # QA:  expected text answer (matched by case-insensitive substring + Vietnamese-aware normaliser)
    # AD_HOC: list of relevant frame_ids; partial credit per scoring formula
    # TRAKE: ordered list of 4 frame_ids
    kis_frame_ids: list[str] | None = None
    qa_answer: str | None = None
    qa_answer_acceptable: list[str] = []     # alternative phrasings
    adhoc_frame_ids: list[str] | None = None
    trake_frame_ids: list[str] | None = None
```

### 3.2 CLI

```
bin/eval [OPTIONS]

Options:
  --tasks PATH           Path to a .jsonl mock-task set
                         (or a directory; all .jsonl files are loaded).
  --system NAME          System version tag (e.g. branch name + short SHA).
  --mode {auto,interactive}   Interactive replays a recorded operator session;
                              auto runs the agent. Default: auto.
  --operator NAME        Required if --mode interactive.
  --output DIR           Output directory. Default: eval-results/<system>/<timestamp>/
  --baseline NAME        Compare against a previous --system result; emit diff.
  --slice EXPR           Run only tasks matching the expression
                         (e.g. "task_type==KIS and difficulty=='hard'").
  --time-budget INT      Override per-task time limit in seconds.
  --concurrency INT      Max concurrent task evaluations. Default: 1.
  --dres-url URL         DRES endpoint. Default: http://localhost:8080
```

### 3.3 Output

```
eval-results/<system>/<timestamp>/
  report.html          # human-readable summary with per-class slicing
  metrics.json         # machine-readable; CI consumes this
  submissions.parquet  # one row per (task, attempted_submission)
  traces/<task_id>.json   # planner trace, tool outputs, timing breakdown
  README.md            # auto-generated provenance: git SHA, env, config
```

## 4. Behaviour

### 4.1 Normal flow
1. CLI loads the task set, validates schema.
2. Boots the backend in Docker (or attaches to a running instance via `--dres-url`).
3. For each task: starts a local DRES task, runs the system, captures every submission, captures latency at each pipeline stage.
4. Polls DRES for scoring decisions in real time.
5. Persists metrics and traces.
6. Generates `report.html` and `metrics.json`.

### 4.2 Failure modes
- **Task schema invalid**: refuse to start; print the offending line.
- **DRES unreachable**: retry 3× with backoff; then fail loudly. Do not silently skip tasks.
- **System backend down**: fail the task with `failure_kind: backend_down`; continue to next task.
- **Wall-clock timeout**: submit the best candidate available; flag `partial_completion: true`.

### 4.3 Determinism
- Random seeds for planner LLM (where applicable) and any stochastic step must be logged. Default: `seed = task_id_hash`.
- Order of task execution is deterministic per run (sorted by `task_id`).

## 5. Acceptance criteria

- **AC1**: `bin/eval --tasks tests/mock_tasks/smoke_20.jsonl --system v0.0.1-baseline` runs to completion against a stub backend and emits `report.html` + `metrics.json`.
- **AC2**: The schema validator rejects a mock-task JSONL with a missing `task_type` field and prints a useful error.
- **AC3**: When DRES is unreachable for >30 seconds, the run fails with exit code 2 and a diagnostic message; no metrics are written.
- **AC4**: `metrics.json` includes all metrics from §6 of the parent proposal and is loadable by `json.load`.
- **AC5**: `report.html` includes a per-task-type slicing table (KIS / QA / AD_HOC / TRAKE) and a per-class slicing table by `metadata.difficulty`.
- **AC6**: Running with `--baseline <previous-system>` adds a diff column to `report.html` and writes per-metric deltas to `metrics.json` under key `delta_vs_baseline`.
- **AC7**: `bin/eval` exit code is non-zero if any of the *pre-registered CI thresholds* (defined in `eval/ci_thresholds.json`) is violated; zero otherwise. The thresholds initially include: KIS R@1 ? 0.50, QA correctness ? 0.40, p95 end-to-end < 2.0 s.
- **AC8**: Determinism — running the same task set twice with the same seed on the same system produces metrics agreeing to ±0.1 % across all numeric fields.

## 6. Non-functional requirements

- **Latency**: harness overhead (excluding the system under test) ? 5 % of task wall-clock.
- **Throughput**: must run the 300-task full set in ? 30 minutes on a single workstation with `--concurrency=1` (sequential), or ? 10 minutes at `--concurrency=4`.
- **Memory**: < 4 GB RAM for the harness itself (excluding the system under test).
- **Compatibility**: Python 3.11+, Docker 24+, Postgres 16 (DRES dependency). Linux primary; Windows acceptable but not gated by CI.

## 7. Dependencies

- **Internal**: none yet (this is the foundation).
- **External**:
  - DRES (Distributed Retrieval Evaluation Server) — <https://github.com/dres-dev/DRES>, used at LSC/VBS.
  - `pydantic >= 2`, `httpx`, `duckdb`, `pyarrow`, `jinja2` (for report rendering), `pytest` (developer tests).
- **Data**:
  - `tests/mock_tasks/smoke_20.jsonl` (20 hand-curated tasks for per-PR CI; included with this spec's first PR)
  - `tests/mock_tasks/full_300.jsonl` (300 tasks; assembled over Phase 1; see Open Questions)

## 8. Test plan

### 8.1 Unit tests (`tests/unit/test_eval_harness.py`)
- `test_schema_rejects_missing_task_type_AC2`
- `test_metrics_json_loadable_AC4`
- `test_baseline_diff_appears_in_report_AC6`
- `test_ci_thresholds_exit_code_AC7`
- `test_determinism_same_seed_same_metrics_AC8`

### 8.2 Integration tests (`tests/integration/test_bin_eval_smoke.py`)
- Boot DRES via testcontainers, point harness at a stub backend that returns canned top-10, run the 20-task smoke set, assert `report.html` exists and `metrics.json` parses.

### 8.3 Manual smoke
- Open the generated `report.html` in a browser; verify the per-class table renders.

## 9. Open questions

- **Q1**: Source for 300 mock tasks. Plan: 50 translated from LSC archive, 50 from AIC HCMC public materials, 100 team-generated, 100 adversarial via Gemini. Authoring this corpus is its own workstream (~1 engineer-week); does it warrant its own SPEC-NNNN? Recommend: yes (`SPEC-0001a` or a fresh ID).
- **Q2**: Should `bin/eval --mode interactive` record operator clicks via a browser-extension or via the React UI's own logging? Recommend the latter (no extension needed), but it pushes a dependency on SPEC-0012.
- **Q3**: CI threshold values in §5 AC7 are placeholders. They must be calibrated against the v0.0.1 baseline before being enforced. Open a follow-up issue to lock the thresholds after Phase 1.
- **Q4**: Concurrency limit. At `--concurrency > 1`, do we share a single backend or spin up replicas? Recommend single backend (the system under test is what we are measuring), with rate-limiting at the harness side.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-26 | team lead | Created (Draft) |
