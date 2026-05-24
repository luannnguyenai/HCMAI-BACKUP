# Proposal 05 - Evaluation Harness

> Without a fast, reliable, repeatable internal evaluation we are guessing. This proposal describes our internal "mock-DRES" + regression test infra.
>
> **Why this is a load-bearing proposal for the original contributions:** the C1 (DiacriticBERT), C2 (learned fusion), and C4 (agent self-distillation) workstreams in [`08-original-contributions.md`](08-original-contributions.md) each ship behind an *ablation gate*. The harness is what runs those ablations. If the harness is weak, the ablations are uninterpretable and we end up shipping or killing contributions on vibes. SS 13-15 below describe the contribution-specific eval extensions.

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
- **C1/C2/C4 ablation reports** (SS 13-15): triggered before each ship/no-ship decision and re-run weekly while the contribution is active.

## 13. C1 - Diacritic-noise robustness eval (`eval/diacritic_robustness.py`)

For DiacriticBERT (proposal 08 SS 3). Two evals:

### 13.1 Synthetic noise sweep
- Take 500 clean Vietnamese queries from the dev set.
- Apply each of the 4 noise modes (`drop_all`, `random_drop_p` with `p in {0.3, 0.5, 0.8}`, `tone_swap_p`, `mixed`) -> ~7000 (clean -> noisy) query instances.
- For each instance: retrieve top-10 against the OCR/ASR/caption index; the gold is the frame retrieved for the *clean* query.
- Metrics: **degradation@10** = `R@10(noisy) / R@10(clean)`. Target: with C1 on, `degradation@10 >= 0.85`; without C1 (BGE-M3 only baseline), expect `~0.65-0.75`.

### 13.2 Real-task slice
- On the 300-task dev set, tag the queries that involve Vietnamese proper nouns or scene-text references (~80-120 queries).
- A/B: report R@1 / R@5 / NDCG@10 with `c1_on=true` vs `c1_on=false`.
- Ship gate: >=1% R@1 lift on this slice AND no class regresses by >1.5%.

### 13.3 Negative-result handling
- If the synthetic sweep shows lift but the real-task slice does not, document the distribution gap in `experiments/c1/RESULT.md` and ship the SeaLLMs-v3 query-rewriting fallback.

## 14. C2 - Learned-fusion vs RRF A/B (`eval/fusion_ablation.py`)

For per-task-type learned fusion (proposal 08 SS 4).

### 14.1 Per-task ablation
- For each task type t in {KIS, QA, TRAKE, Ad-hoc}: on the 75-query slice for that type, report NDCG@10 for 3 fusion modes:
  1. RRF k=60 (pre-C2 baseline and runtime auto-fallback)
  2. Single global learned model (no per-task split)
  3. Per-task-type learned model (C2 - the proposed new default)
- Statistical significance: bootstrap 1000 resamples; report 95% CI. Ship gate: mode 3 beats mode 1 by >=1 point with non-overlapping CI on >=3 of 4 task types.

### 14.2 Leakage check
- Leave-one-task-out CV: train on 75% of tasks per type, test on held-out 25%. The gap between train and test NDCG@10 must be <3 points; otherwise we have overfit the dev set.

### 14.3 Runtime guardrail validation
- Inject 50 known-bad queries (gold not in top-200) and confirm the runtime auto-fallback to RRF triggers within the streaming 50-query window.

## 15. C4 - Agent self-distillation ablation (`eval/agent_automatic_ablation.py`)

For planner self-distillation (proposal 08 SS 6).

### 15.1 Pre-distillation baseline
- Before any DSPy round: run the zero-shot SeaLLMs-v3-7B planner on the automatic-track mock-finals (30 queries). Record per-task-type R@1 / R@5, time-to-submit, Brier-calibrated confidence.

### 15.2 Post-distillation eval
- After each DSPy MIPRO round: re-run on the same frozen 30-query set + a 70-query held-out set (never seen by DSPy).
- Ship gate: held-out R@1 must beat the previous prompt by >=2% with paired bootstrap significance.

### 15.3 Continual-refresh quality monitor
- After each prelim and mock-finals batch, re-run 15.2. Plot R@1 over time. If R@1 plateaus or regresses, freeze the corpus and inspect for trace pollution (e.g., wrong-submission traces accidentally included).

## 16. VLM rerank position-bias sweep (`eval/rerank_position_bias.py`)

Applies to all VLM-as-judge configurations (proposal 01 SS 5.9 + C5 in proposal 08 SS 7).

### 16.1 Position-bias quantification
- Take 100 dev-set queries. For each, render the same 9 candidates in 6 different grid positions (rotate + transpose). Record the rank assigned to the gold candidate in each position.
- Metric: **position-variance** = std-dev of gold-rank across the 6 positions. Lower is better.
- Baseline (single-shot direct ranking): expect position-variance ~1.8-2.5 ranks.
- With 3-vote input-shuffle mitigation: target <=1.0.
- With C5 counterfactual pruning: target <=0.5.

### 16.2 Cost-vs-quality curve
- Plot rerank R@1 (y-axis) vs rerank latency (x-axis) for {1-vote direct, 3-vote shuffle, C5 counterfactual}. Use this curve in the finals slide deck to justify the chosen mode per task type.
