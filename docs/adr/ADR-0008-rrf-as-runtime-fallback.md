---
id: ADR-0008
title: RRF k=60 is the runtime fallback; C2 learned fusion is the default
status: Accepted
decided_on: 2026-05-24
deciders:
  - team lead
related_adrs:
  - ADR-0007
---

# ADR-0008 — RRF k=60 is the runtime fallback; C2 learned fusion is the default

## Status

Accepted.

## Context

The first draft of [`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) §2.4 specified **Reciprocal Rank Fusion (RRF, Cormack 2009)** with `k=60` as the default fusion across 12–15 heterogeneous ranked lists (4 image-text dense × 3 Vietnamese paraphrases + 9 lexical/sparse text lists ± an optional CLAP audio-event list).

RRF is robust because it ignores score magnitudes. By the same token, it ignores ranker *quality*. We stack ranked lists from very different distributions across four task types (KIS / QA / TRAKE / Ad-hoc), where the optimal weighting differs by task — KIS rewards image-text dense, QA leans on OCR/caption, Ad-hoc is the wild card.

ADR-0007 makes per-task-type learned fusion (C2) a primary original contribution. This ADR captures the consequent inversion of the default-vs-fallback relationship.

## Decision

The **default** runtime fusion is **C2** — a **LightGBM LambdaRank** model selected at query time by the planner LLM's emitted `task_type` field. There is one LambdaRank model per task type (4 models total).

**RRF k=60** is retained as the **runtime fallback**, activated automatically when any of:

1. The planner failed to emit a recognised `task_type`.
2. The selected LambdaRank model has not yet been trained (cold start in Phase 1).
3. A held-out slice in the eval harness reports the LambdaRank model regresses against RRF for the current task type (CI gate).

The fallback is logged and counts towards a CI metric (`fusion_fallback_rate`); a spike triggers investigation.

## Consequences

### Positive
- Captures the lift from learned ranker weighting (expected +1–3 % NDCG@10) while preserving RRF's robustness as an always-available safety net.
- The fallback is invisible to the operator — if the learned model breaks at finals, the system silently degrades to a known-good baseline rather than failing.
- Makes the contribution measurably ship-or-ship-not via the CI gate; no judgment calls at deploy time.

### Negative
- Two fusion code paths to maintain. The LightGBM models become a versioned artifact (re-trained per task type, calibrated against per-task held-out slices).
- Cold-start period in Phase 1 runs on RRF, so any architecture decisions tested before C2 ships use RRF — must be re-validated post-C2.

### Neutral / observable
- Score-distribution-aware normalisation (min-max per ranker) is required for LambdaRank features; RRF doesn't need this. The wiring adds a small normalisation step always-on.

## Alternatives considered

- **Keep RRF as default; ship C2 as opt-in** — safer but loses the lift in 90 %+ of well-trained cases — rejected because the auto-fallback gate already gives us safety without the opt-in tax.
- **Pure learned fusion, no RRF fallback** — cleaner architecture — rejected because the cold-start period in Phase 1 and any per-task LambdaRank failure leaves the system without a fusion at all.
- **Weighted RRF (per-source weights tuned globally, not per-task-type)** — middle ground — rejected because per-task-type variation is the empirical observation we're trying to capture; a global weight regresses on the worst-fit task.

## References

- Cormack, Clarke, Buettcher (2009) "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods" — original RRF paper
- LightGBM LambdaRank — <https://lightgbm.readthedocs.io/en/latest/Features.html#lambdarank>
- [`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md) §4 (C2 method spec)
- [`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) §2.4 (to be updated)
