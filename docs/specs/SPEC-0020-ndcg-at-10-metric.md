---
id: SPEC-0020
title: NDCG@10 ranking metric in the evaluation harness
status: Implemented
owner: unassigned
created: 2026-05-29
updated: 2026-05-29
implements_proposal: docs/proposals/05-evaluation-harness.md SS 14
related_adrs:
  - ADR-0007
  - ADR-0008
depends_on:
  - SPEC-0001
---

# SPEC-0020 - NDCG@10 ranking metric in the evaluation harness

> Adds Normalised Discounted Cumulative Gain at rank 10 (NDCG@10) to the per-task and aggregate metrics emitted by `bin/eval`. NDCG@10 is the ship-gate metric for the C2 per-task-type learned fusion contribution (ADR-0007, ADR-0008; proposal 05 SS 14). SPEC-0001's metric set (R@1/5/10, MRR, latency) does not include it, and SPEC-0001's acceptance criteria are frozen; this spec adds NDCG@10 as a small, separable unit on top of the SPEC-0001 harness.

## 1. Context

[`docs/proposals/05-evaluation-harness.md`](../proposals/05-evaluation-harness.md) SS 14 defines the C2 learned-fusion ablation gate in terms of **NDCG@10**: "mode 3 (per-task learned) beats mode 1 (RRF k=60) by >=1 point NDCG@10 with non-overlapping bootstrap CI on >=3 of 4 task types." [ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md) and [ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md) make C2 a primary contribution whose expected lift is stated as "+1-3% NDCG@10".

The harness implemented under SPEC-0001 currently emits R@1/R@5/R@10, MRR, KIS/Ad-hoc scores, wrong-submission rate, and latency - but not NDCG@10. Without it, the C2 ablation (future SPEC-0015) cannot be measured against its own gate, and the metric cannot be cited in the post-competition paper. This spec closes that gap at the harness layer so the metric is available to every consumer (C1/C2/C4 ablations, nightly regression, paper tables).

## 2. Scope

### 2.1 In scope
- A pure `ndcg_at_k(submissions, correct_ids, k)` function (and its `dcg_at_k` helper) in `aic2026.harness.scoring`, using **binary relevance** gains (the harness only has binary ground truth, not graded relevance).
- Wiring NDCG@10 into `score_task` for all four task types (KIS / QA / AD_HOC / TRAKE), consistent with how R@k is already handled per type.
- A per-task `ndcg_at_10` field on `TaskMetrics` and a `mean_ndcg_at_10` field on `TaskTypeAggregate`, emitted to `metrics.json`.
- NDCG@10 surfaced in `report.html` (overall row + by-task-type column).
- A `metrics.json` `schema_version` bump from `"1"` to `"2"` (additive but the strict, `extra="forbid"` models gain required fields).

### 2.2 Out of scope
- Graded (multi-level) relevance. The harness ground truth is binary; graded NDCG is deferred until/if a graded corpus exists.
- The C2 fusion ablation logic itself (that is SPEC-0015; this spec only provides the metric it will consume).
- Bootstrap confidence intervals and significance testing (separate follow-up; proposal 05 SS 14.1 names them but they are not gated here).
- NDCG at other cutoffs (NDCG@5, NDCG@20). Only `@10` is required by the C2 gate; the helper is parameterised by `k` but only `@10` is emitted.

## 3. API contract / interface

```python
# aic2026/harness/scoring.py

def dcg_at_k(submissions: list[Submission], correct_ids: set[str], k: int) -> float:
    """Discounted cumulative gain at k with binary gains.

    DCG@k = sum_{i=1..k} rel_i / log2(i + 1), where rel_i is 1 if the i-th
    submission's frame_id is in `correct_ids`, else 0. Submissions are assumed
    sorted by rank ascending (rank 1 first).
    """

def ndcg_at_k(submissions: list[Submission], correct_ids: set[str], k: int) -> float:
    """Normalised DCG at k with binary gains.

    NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the ideal ranking
    (all relevant items first): IDCG@k = sum_{i=1..min(|correct_ids|, k)} 1/log2(i+1).
    Returns 0.0 when there are no relevant items or IDCG@k == 0.
    Raises ValueError when k <= 0.
    """
```

`score_task(...)` gains one output-dict key, `"ndcg_at_10"` (float in [0, 1]).

```python
# aic2026/models/metrics.py  (additions)

class TaskMetrics(BaseModel):
    ...
    ndcg_at_10: float = Field(ge=0.0, le=1.0)   # new, after mrr

class TaskTypeAggregate(BaseModel):
    ...
    mean_ndcg_at_10: float = Field(ge=0.0, le=1.0)   # new, after mean_mrr

class AggregateMetrics(BaseModel):
    schema_version: str = "2"   # bumped from "1"
```

## 4. Behaviour

NDCG@10 uses binary gains because ground truth is binary (`correct_ids` is a set membership test).

- **KIS** (single relevant frame): `correct_ids` is the set of acceptable frame_ids. NDCG@10 = `1 / log2(r + 1)` when the correct frame is at rank `r <= 10` (IDCG = `1/log2(2)` = 1), else 0. This is more informative than the binary R@10 because it rewards a higher rank.
- **AD_HOC** (multiple relevant frames): standard binary NDCG@10 over the relevant pool; IDCG normalises by `min(|relevant|, 10)`.
- **TRAKE**: computed over membership of the 4 ground-truth frames in the top-10 (order-insensitive), consistent with how `r_at_k` is already computed for TRAKE. The order-sensitive correctness remains `trake_correct` / `ok`; NDCG@10 is a secondary signal only.
- **QA** (no frame ranking): collapses to `{0.0, 1.0}` equal to QA correctness, mirroring how `r_at_1/r_at_5/r_at_10/mrr` already collapse for QA in `score_task`.
- **Empty relevant set**: NDCG@10 = 0.0 (never raises).
- **k <= 0**: `ndcg_at_k` raises `ValueError` (matches `r_at_k` contract).
- **Items beyond k**: ignored (only the top-k contribute to DCG).

Determinism: pure function of (submissions, correct_ids, k); no I/O, no randomness. Same inputs -> bitwise-same output.

## 5. Acceptance criteria

- **AC1**: For a single relevant item, `ndcg_at_k` returns `1.0` when it is at rank 1, and `1 / log2(r + 1)` when at rank `r <= k`; for a perfect multi-relevant ranking (all relevant items first) it returns `1.0`. Verified in `tests/unit/test_scoring.py`.
- **AC2**: `metrics.json` produced by a smoke run carries `ndcg_at_10` on every per-task row and `mean_ndcg_at_10` on every `by_task_type` bucket and on `overall`, and the file round-trips through `AggregateMetrics.model_validate`. The file's `schema_version` is `"2"`. Verified in `tests/unit/test_ndcg_metrics_AC2.py`.
- **AC3**: `ndcg_at_k` returns `0.0` for an empty relevant set, ignores items ranked beyond `k`, and raises `ValueError` for `k <= 0`. Verified in `tests/unit/test_scoring.py`.
- **AC4**: For QA tasks, the emitted `ndcg_at_10` equals the QA correctness value (`1.0` if correct else `0.0`). Verified in `tests/unit/test_scoring.py` via `score_task`.

## 6. Non-functional requirements

- **Latency**: NDCG@10 adds O(k) work per task (k=10); negligible (< 1 ms/task) and well within the SPEC-0001 SS 6 harness-overhead budget of <=5% of task wall-clock.
- **Compatibility**: Python 3.11+. No new runtime dependencies (`math.log2` from stdlib).
- **Accuracy**: NDCG values agree with hand-computed references to within 1e-9 in unit tests.

## 7. Dependencies

- **Internal**: SPEC-0001 (provides `Submission`, `MockTask`, `score_task`, `TaskMetrics`, `TaskTypeAggregate`, the runner, and the HTML/JSON reporters that this spec extends).
- **External**: none new (`math` from the standard library).
- **Data**: reuses `tests/mock_tasks/smoke_20.jsonl` for the AC2 smoke test.

## 8. Test plan

- **Unit tests** (`tests/unit/test_scoring.py`):
  - `test_ndcg_perfect_ranking_is_one_AC1`
  - `test_ndcg_single_relevant_rank_discount_AC1`
  - `test_ndcg_partial_multi_relevant_AC1`
  - `test_ndcg_empty_relevant_is_zero_AC3`
  - `test_ndcg_ignores_items_beyond_k_AC3`
  - `test_ndcg_invalid_k_raises_AC3`
  - `test_score_task_qa_ndcg_collapses_to_correctness_AC4`
- **Unit tests** (`tests/unit/test_ndcg_metrics_AC2.py`):
  - `test_ndcg_present_in_metrics_json_AC2` (smoke run -> per-task + aggregate keys present, schema_version "2", round-trips)
- **Eval-harness** (`bin/eval --tasks tests/mock_tasks/smoke_20.jsonl --system <branch>`):
  - Confirms `report.html` renders the NDCG@10 column and `metrics.json` carries the field end-to-end.

## 9. Open questions

- **Q1**: Graded relevance. If a future corpus assigns relevance levels (e.g. partially-relevant Ad-hoc frames), `dcg_at_k` should switch from binary gains to `2^rel - 1`. Deferred until such a corpus exists; flagged here so the change is anticipated.
- **Q2**: Should the C2 gate (SPEC-0015) also report NDCG@5 for sensitivity analysis? Not required by ADR-0007; the helper is k-parameterised so it is cheap to add later.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-29 | implementer (user-directed) | Created; Draft -> Approved -> Implementing in one pass for solo work per CONTRIBUTING SS "The workflow". Implementation PR opens against branch `spec/0020-ndcg-at-10-metric`. |
| 2026-05-29 | implementer | Merged in PR #5; status -> Implemented. |
