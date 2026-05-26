---
id: ADR-0007
title: Three primary original contributions — C1 DiacriticBERT, C2 learned fusion, C4 agent self-distillation
status: Accepted
decided_on: 2026-05-24
deciders:
  - team lead
related_adrs:
  - ADR-0001
---

# ADR-0007 — Three primary original contributions: C1, C2, C4

## Status

Accepted.

## Context

ADR-0001 reframed the strategy as Floor / Edge / Moat. The Edge tier requires concrete, original technical contributions on top of the reproduced 2026 SOTA floor. [`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md) scopes five candidates (C1–C5); this ADR commits us to which are *primary* (must ship by Phase 2 decision gate) and which are *backup* (ship only if Phase 2 has slack).

## Decision

Three contributions are **primary** — they must each have a tested ablation result by the Phase 2 ? Phase 3 decision gate, and at least two must pass to satisfy the gate:

- **C1 — DiacriticBERT**: a diacritic-robust late-interaction head over frozen BGE-M3, trained on a controlled Vietnamese diacritic-noise schedule. Attacks the systematic ASR/OCR failure mode (master strategy §7 item 3). Expected lift: +2–5 % R@1 on OCR/ASR-bridged queries. Owner: Vietnamese NLP Engineer. Effort: ~1 week.

- **C2 — Per-task-type learned fusion**: a LightGBM LambdaRank model selected at query time by the planner's emitted `task_type` field, replacing uniform RRF k=60. Runtime auto-fallback to RRF if the learned model regresses on a held-out slice. Expected lift: +1–3 % NDCG@10. Owner: Lead Engineer. Effort: ~3 days. Also captured by ADR-0008.

- **C4 — Agent self-distillation**: the interactive-track operator's correct, fast submissions become the training corpus for the automatic-track planner via DSPy MIPRO. The trace logger is wired into the interactive system from Phase 1 so the corpus accumulates throughout development. Expected lift: +10–20 % automatic-track R@1 vs zero-shot planner. Owner: Operator-1 / ML Engineer. Effort: ~1 week after interactive system stabilises.

Two contributions are **backup** — ship only if Phase 2 has slack:

- **C3 — PriorDP**: story-graph generalisation of DANTE for TRAKE. ~2 weeks.
- **C5 — Counterfactual VLM rerank**: iterative pruning rerank for OOK named entities. ~1 week.

Each primary contribution is independent — any one can ship without the others — and additive — the dev-set ablation reports each alone and stacked.

## Consequences

### Positive
- Concrete, measurable bets we can defend in a finals Q&A and write up as a post-competition paper.
- Each attacks a known failure mode documented in our research notes — not novelty for novelty's sake.
- Bounded scope (head-not-backbone for C1, single LightGBM model for C2, prompt-not-training for C4) keeps risk manageable.

### Negative
- Phase 2 budget tightens: 4 weeks must accommodate floor fine-tunes + three contribution workstreams.
- If none of C1/C2/C4 pass, we ship only the floor — the strategy's "honest summary" explicitly allows this but the team should expect a tighter finals race.

### Neutral / observable
- The eval harness ([`docs/proposals/05-evaluation-harness.md`](../proposals/05-evaluation-harness.md)) gains gate logic for the C1/C2/C4 ablations.
- Specs SPEC-0014, SPEC-0015, SPEC-0016 are reserved in [`docs/specs/INDEX.md`](../specs/INDEX.md) for these three contributions.

## Alternatives considered

- **Make C3 PriorDP primary** — TRAKE is high-value — rejected because TRAKE inclusion in 2026 is not yet confirmed, and C3 has ~2-week effort with no shippable fallback if the technique fails.
- **Make C5 counterfactual rerank primary** — OOK entities are a real failure mode — rejected because Vintern is a smaller VRAM budget on the 5070 (ADR-0003) and counterfactual rerank requires multiple VLM passes per query; bakeoff-pending feasibility.
- **Add a fourth primary (e.g. fine-tune ColVintern)** — more shots on goal — rejected because three concurrent novelty workstreams already saturates a 5-person team in Phase 2.

## References

- [`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md)
- [`docs/strategy/00-master-strategy.md`](../strategy/00-master-strategy.md) §2.2 and §4 Phase 2
- ADR-0008 (C2 inversion of fusion default)
