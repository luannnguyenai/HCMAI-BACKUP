---
id: ADR-0002
title: We describe the stack as "Vietnamese-capable", not "Vietnamese-native"
status: Accepted
decided_on: 2026-05-24
deciders:
  - team lead
---

# ADR-0002 — "Vietnamese-capable" not "Vietnamese-native"

## Status

Accepted.

## Context

The first draft of the strategy called our model stack "Vietnamese-native." The audit underlying ADR-0001 flagged this as inaccurate:

- **Meta CLIP 2 ViT-H/14** is *multilingual from scratch* — strong on Vietnamese (XM3600 I?T 64.3) but not Vietnamese-first.
- **SigLIP-2** is multilingual.
- **BGE-M3** is multilingual (170+ languages).
- Only **Vintern** (5CD-AI) and **PhoWhisper** (VinAI) and **VietOCR** (pbcquoc) are genuinely Vietnamese-first.

Claiming "Vietnamese-native" overstates our differentiation. A reviewer who knows Meta CLIP 2 will dismiss the claim and, by extension, weaken trust in everything else we say.

## Decision

We describe the stack as **"Vietnamese-capable"** throughout the strategy, proposals, illustrations, and press materials. Where we cite specific Vietnamese-first components, we name them: Vintern-3B-beta, PhoWhisper-large, VietOCR, ColVintern (when in scope).

## Consequences

### Positive
- Survives technical audit by anyone who has read the underlying model cards.
- Forces precision when we *do* claim Vietnamese-first capability (only for Vintern/PhoWhisper/VietOCR).

### Negative
- Slightly less marketable in a 30-second pitch. Acceptable cost.

### Neutral / observable
- All documents updated in commit `2023db0`. New documents must use the corrected framing.

## Alternatives considered

- **Keep "Vietnamese-native"** — better marketing — rejected because indefensible under audit.
- **Drop all language-specific claims** — call it "multilingual" — rejected because PhoWhisper/Vintern/VietOCR are genuinely Vietnamese-first and that's a real advantage we should claim.

## References

- [`docs/strategy/00-master-strategy.md`](../strategy/00-master-strategy.md) §2.1 (post-rewrite)
- [`docs/research-notes/04-vietnamese-stack-and-agents.md`](../research-notes/04-vietnamese-stack-and-agents.md) — model-by-model audit of Vietnamese-first vs multilingual
- Meta CLIP 2 paper — <https://arxiv.org/html/2507.22062v3>
