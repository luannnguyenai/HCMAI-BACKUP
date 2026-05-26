---
id: ADR-0001
title: Strategy framed as Floor / Edge / Moat
status: Accepted
decided_on: 2026-05-24
deciders:
  - team lead
---

# ADR-0001 — Strategy framed as Floor / Edge / Moat

## Status

Accepted.

## Context

The first draft of [`docs/strategy/00-master-strategy.md`](../strategy/00-master-strategy.md) presented our winning thesis as "four stacked advantages." A May 24 audit found that 23 of ~25 components in proposals 01–07 were off-the-shelf 2026 SOTA reuse, and each of the four advantages was explicitly attributable to a prior LSC / VBS / AIC winner (MEMORIA, NII-UIT, SnapMind, PraK). Every serious team will read the same papers and converge on a near-identical stack. As a competition plan, reproducing the SOTA is necessary. As a defensible narrative in a finals Q&A, it is not differentiating.

## Decision

We adopt a three-tier framing of the strategy:

- **Floor** — the reproduced 2026 SOTA stack (proposals 01–07). Necessary; not differentiating. We will not market it as our edge.
- **Edge** — three original contributions defined in [`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md): **C1 DiacriticBERT**, **C2 per-task-type learned fusion**, **C4 agent self-distillation**. Two backups: C3 PriorDP, C5 counterfactual VLM rerank. ADR-0007 captures the primary three.
- **Moat** — process advantages: operator drills + submission-verification panel. Every team can do this; few will.

The honest summary: if at least 2 of {C1, C2, C4} pass their dev-set ablations, we are a technically differentiated finalist. If 0 pass, we are a competent floor-only team — still a finalist, but expect a tighter race.

## Consequences

### Positive
- Removes overclaim that was indefensible in finals Q&A.
- Forces explicit, measurable bets (C1/C2/C4) with ablation gates in [`docs/proposals/05-evaluation-harness.md`](../proposals/05-evaluation-harness.md).
- Aligns the team narrative with what actually wins at LSC/VBS (per the SOTA review).

### Negative
- Press-kit / slide-deck story is less marketable than "Vietnamese-native ensemble wins" — but more credible.
- Reading list grows: proposal 08 jumps to second priority in the team's reading order.

### Neutral / observable
- Phase 2 decision gate now requires "at least 2 of {C1, C2, C4} pass ablations" in addition to baseline-beating.
- C2 inverts the default-vs-fallback for fusion (see ADR-0008).

## Alternatives considered

- **Keep the four-advantage framing** — easier slide-deck story — rejected because indefensible under audit and not aligned with empirical LSC evidence.
- **Add advantages, not contributions** — pile on more models — rejected because LSC review §IV-D shows single-model + ensemble plateaued in 2023; only original work moves the curve.
- **Floor only, skip the contributions** — lower-risk timeline — rejected because we then have no story for the finals Q&A and no path to a SoICT special-session paper.

## References

- [`docs/strategy/00-master-strategy.md`](../strategy/00-master-strategy.md) §2 (rewritten in commit `2023db0`)
- [`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md)
- LSC 2022–24 SOTA review — <https://arxiv.org/abs/2506.06743> §IV-D
