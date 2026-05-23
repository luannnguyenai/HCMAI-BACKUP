# Proposal 05 - Evaluation Harness

> Without a fast, reliable, repeatable internal evaluation we are guessing. This proposal describes our internal "mock-DRES" + regression test infra.

## 1. Why a custom harness

The official AIC2026 scoring server (likely DRES-based) will only be available at the preliminary round. We need a local stand-in *now* so we can:
- A/B compare model swaps
- Detect regressions when refactoring
- Train new operators
- Validate the agent's confidence calibration
- Sweep hyperparameters (RRF k, lambda for DANTE, VLM rerank size)

## 2. Mock-DRES service

We host our own **DRES** instance locally. DRES is open-source: <https://github.com/dres-dev/DRES>.

Components:
- DRES server (JVM, Postgres backing store)
- DRES web UI (for managing tasks)
- Custom seeder that loads our mock task definitions

Setup:
```bash
docker-compose -f infra/dres/docker-compose.yml up -d
```

This gives us the exact API contract teams will hit at the preliminary round. Our backend submits via the official DRES client SDK.

## 3. Mock task definitions

Three task buckets, each with 100 hand-curated queries:

### KIS bucket
- Single image as ground truth.
- Vietnamese natural-language description.
- Optional 30-second incremental hints.

### QA bucket
- Vietnamese natural-language question.
- Expected text answer.
- Optional list of supporting frames.

### TRAKE bucket
- 4-scene Vietnamese description (each scene = 1-2 sentences).
- Ground truth = 4 specific frames in correct temporal order.

### Sources for queries
1. **From LSC public archive**: translate ~50 KIS queries from LSC'22-24 into Vietnamese.
2. **From AIC HCMC public materials**: scrape past competition task examples.
3. **Team-generated**: each engineer writes 10 queries per week against their development data; cross-review for quality.
4. **Adversarial**: generate hard cases with Gemini 2.5 Pro asking for "queries that would trick a CLIP-based retriever".

## 4. Metrics tracked

For each (system_version, task_bucket) pair we record:

| Metric | Why |
|---|---|
| Recall@1 / @5 / @10 | Core retrieval quality |
| Mean reciprocal rank | Sensitive to top-1 placement |
| Time-to-first-correct (KIS) | Critical for 50*t/T penalty |
| Wrong submission rate | Critical for 10-pt penalty |
| Per-class recall (sliced by place/ADL/time-of-day) | Detect regressions |
| Confidence calibration (Brier score) | Agent track quality |
| End-to-end latency p50/p95 | UX matters |

All metrics emitted as Prometheus metrics + persisted to DuckDB for ad-hoc analysis.

## 5. Test runner

`bin/eval` is a CLI that:
1. Loads a task set.
2. Spins up our backend in a Docker compose.
3. Runs each task either interactively (with a recorded operator session replayed) or autonomously (via the agent).
4. Submits to local DRES.
5. Pulls scores; writes a report.

```
$ bin/eval --tasks tests/kis_v1.jsonl --system v0.7.2 --mode automatic
Running 100 tasks against system v0.7.2 in automatic mode...
[####################] 100/100 (1m43s)

R@1     0.71  (+0.04 vs v0.7.1)
R@5     0.89  (+0.02)
R@10    0.94  ( 0.00)
MRR     0.79  (+0.03)
TTC     38s   (-12s)
WR%     2.0%  (-0.5%)
Latency p50: 720ms   p95: 1.8s

PASS gate: r@1 > 0.65 ?
PASS gate: latency p95 < 2s ?
```

## 6. CI integration

- Every PR runs the **smoke set** (20 fixed tasks) on GitHub Actions or Gitea Actions.
- Nightly runs the **full set** (300 tasks).
- Regressions >2% in any metric block merge.

## 7. Operator drill mode

`bin/eval --mode interactive --operator alice` mode:
- Spawns the React UI in a controlled headless setting.
- Plays back a recorded query stream into the UI.
- Records the operator's clicks/timings.
- Compares to the gold submission.
- Outputs operator-specific report: avg time-to-submit, mistake rate, hesitation points.

This lets us train operators rigorously, not just "play around with the system".

## 8. A/B harness for model changes

`bin/abeval`:
- Takes two system versions A and B.
- Runs both on the same task set.
- Outputs per-task delta + statistical significance (paired bootstrap).
- Flags tasks where B regresses vs A so we can debug.

Critical for fine-tuning experiments: never ship a model if any task class regresses by more than 2%.

## 9. Confidence calibration evaluation

Specifically for the automatic agent:
1. For each task, record the agent's confidence score on its top submission.
2. Bucketize into deciles.
3. Plot mean accuracy vs mean confidence -> calibration curve.
4. Brier score = mean squared error between confidence and 0/1 correctness.
5. Target Brier < 0.15.

If calibration is bad, the agent submits low-confidence answers and incurs the 10-pt penalty. We will spend 1 week tuning this in Phase 2.

## 10. Reproducibility

Every eval run must:
- Pin the system version (git SHA).
- Pin the task set (file SHA).
- Record the exact env: model versions, hyperparams.
- Output to `eval-results/<run_id>/report.html` + `metrics.json` + `submissions.parquet`.

Long-term: we can re-run any historical eval and reproduce numbers within +/-1%.

## 11. The single most important metric

The **time-weighted KIS score** is our north star:
```
score = (0.5 * R@1 + 0.5 * MRR) * exp(-time_seconds / 300)
```

Maximising this means we maximise correctness AND speed. KIS speed at finals is what historically separates winners.

## 12. Frequency

- **Per-PR smoke** (20 tasks): always.
- **Nightly full** (300 tasks): always.
- **Weekly leaderboard** posted to team Slack.
- **Pre-phase-gate** (Phases 1-4): full set + per-class report + operator drill report.
