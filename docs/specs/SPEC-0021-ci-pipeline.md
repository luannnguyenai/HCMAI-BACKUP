---
id: SPEC-0021
title: Continuous integration pipeline (lint + test + smoke-eval gate)
status: Implementing
owner: unassigned
created: 2026-05-29
updated: 2026-05-29
implements_proposal: docs/proposals/05-evaluation-harness.md SS 6
related_adrs:
  - ADR-0009
depends_on:
  - SPEC-0001
---

# SPEC-0021 - Continuous integration pipeline (lint + test + smoke-eval gate)

> A GitHub Actions workflow that runs on every pull request to `main` and every push to `main`: it formats-checks, lints, unit-tests, and runs the 20-task smoke evaluation. This makes the "numbers, not vibes" discipline in CONTRIBUTING enforceable rather than aspirational, and would have caught the `bin/eval` executable-bit regression automatically.

## 1. Context

[`CONTRIBUTING.md`](../../CONTRIBUTING.md) states "Every PR runs the smoke set (20 fixed tasks) on GitHub Actions or Gitea Actions" and [`docs/proposals/05-evaluation-harness.md`](../proposals/05-evaluation-harness.md) SS 6 makes the per-PR smoke run a gate. [SPEC-0001](SPEC-0001-evaluation-harness.md) SS 2.1 lists "Per-PR smoke set (20 fixed tasks); nightly full set (300 tasks)" in scope and AC7 reserves CI-threshold gating for Tier 3. Today there is **no `.github/workflows/` directory** - nothing runs automatically, so a contributor can merge a PR that fails `ruff` or `pytest`. The `bin/eval` mode bug (committed `100644`, fixed in PR #5) is a concrete example of a regression a smoke gate would have caught.

This spec defines the minimum viable CI gate. It deliberately does **not** enforce score thresholds yet: the harness runs against a deterministic stub backend (real retrieval lands with SPEC-0004/0006/0007), so score-threshold gating stays deferred to SPEC-0001 AC7 with the placeholder thresholds in SPEC-0001 SS 9 Q3. The CI smoke step is a "does it run" gate, not a "is it good" gate.

## 2. Scope

### 2.1 In scope
- A single workflow file `.github/workflows/ci.yml`.
- Triggers: `pull_request` targeting `main`, and `push` to `main`.
- Steps, all failing the job on non-zero exit: dependency sync (`uv sync --frozen`), `ruff format --check`, `ruff check`, `pytest`, and a 20-task smoke run via `./bin/eval`.
- Dependency + Python caching for speed.
- Concurrency control: cancel superseded in-progress runs on the same ref.

### 2.2 Out of scope
- Nightly full 300-task run (separate scheduled workflow; tracked under SPEC-0001 SS 2.1).
- Score-threshold enforcement / regression gating against a baseline (SPEC-0001 AC7, Tier 3; needs a real backend and calibrated thresholds).
- Multi-OS / multi-Python matrix (Linux + 3.11 only; matches SPEC-0001 SS 6 "Linux primary").
- Branch-protection rules (a GitHub repo-settings concern, configured in the UI, not in this file).
- Deploy / release automation.

## 3. API contract / interface

Not a code module; the "interface" is the workflow trigger + the gate contract:

```yaml
# .github/workflows/ci.yml (shape, not full content)
on:
  pull_request: { branches: [main] }
  push: { branches: [main] }
jobs:
  lint-test-smoke:
    runs-on: ubuntu-latest
    steps: [checkout, install uv, uv sync --frozen,
            ruff format --check, ruff check, pytest,
            ./bin/eval --tasks tests/mock_tasks/smoke_20.jsonl --system ci-<sha>]
```

The job exit code is the contract: **0 iff** format-check, lint, all tests, and the smoke run each exit 0.

## 4. Behaviour

- **PR opened / updated against `main`**: the workflow runs; a failing step fails the whole job and surfaces a red check on the PR.
- **Push to `main`** (e.g. a merge): the workflow runs as a post-merge guard.
- **Formatting drift**: `ruff format --check .` exits non-zero if any file is not already formatted; the job fails (contributor must run `ruff format`).
- **Lint failure**: `ruff check .` non-zero -> job fails.
- **Test failure**: `pytest` non-zero -> job fails.
- **Smoke-eval failure**: `./bin/eval ...` non-zero (harness crash, schema break, or - as a side effect - a non-executable `bin/eval`) -> job fails. Score values are not asserted at this tier.
- **Concurrency**: a newer commit on the same branch cancels the older in-progress run.

## 5. Acceptance criteria

- **AC1**: The workflow triggers on `pull_request` to `main` and on `push` to `main`. (Verified by inspection of the `on:` block and by the check appearing on this PR.)
- **AC2**: The job runs `ruff format --check .`, `ruff check .`, and `pytest`, and a failure in any one fails the job. (Verified by the green check on a clean tree; locally mirrored by `uv run ruff format --check . && uv run ruff check . && uv run pytest -q`.)
- **AC3**: The job runs the 20-task smoke set via `./bin/eval` and fails if it exits non-zero. Score thresholds are not asserted (deferred to SPEC-0001 AC7). (Verified by the smoke step exit code.)
- **AC4**: A clean checkout of `main` passes the full pipeline (green check). (Verified once this PR's check runs.)

## 6. Non-functional requirements

- **Latency**: total job wall-clock < 10 minutes on `ubuntu-latest` (timeout set to 10 min); with dependency cache, target < 3 minutes.
- **Cost**: GitHub-hosted `ubuntu-latest` minutes only; no self-hosted runner.
- **Determinism**: `uv sync --frozen` uses the committed `uv.lock`; the smoke run uses the fixed seed in the harness (SPEC-0001 SS 4.3).
- **Compatibility**: Python 3.11 (from `.python-version`), `uv` latest, Ubuntu latest.

## 7. Dependencies

- **Internal**: SPEC-0001 (provides `bin/eval`, the `eval` console script, and `tests/mock_tasks/smoke_20.jsonl`).
- **External**: GitHub Actions; `actions/checkout`, `astral-sh/setup-uv`. No new Python dependencies.
- **Data**: `tests/mock_tasks/smoke_20.jsonl` (already in-repo).

## 8. Test plan

- **Self-validating**: the PR that adds `ci.yml` is itself the first run; a green check on this PR proves AC1-AC4.
- **Local mirror** (what CI runs, runnable by any contributor):
  - `uv sync --frozen`
  - `uv run ruff format --check .`
  - `uv run ruff check .`
  - `uv run pytest -q`
  - `./bin/eval --tasks tests/mock_tasks/smoke_20.jsonl --system "local-smoke"`
- **Negative check** (manual, optional): introduce a formatting error on a scratch branch and confirm the job goes red.

## 9. Open questions

- **Q1**: Nightly full-300 run - schedule (`cron`) in this same file or a separate workflow? Recommend separate (`nightly.yml`) once the 300-task corpus exists (SPEC-0001 Q1). Not gated here.
- **Q2**: When SPEC-0001 AC7 lands (`eval/ci_thresholds.json`), the smoke step should switch from "does it run" to "meets thresholds". That is a one-line change to the smoke step plus a `--baseline` flag; tracked under SPEC-0001, not here.
- **Q3**: Branch protection (require this check before merge) is a repo-settings action the repo owner must take in the GitHub UI; flagged for the owner, out of scope for the file.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-29 | implementer (user-directed) | Created; Draft -> Approved -> Implementing in one pass for solo work per CONTRIBUTING. Implementation PR opens against branch `spec/0021-ci-pipeline`. |
