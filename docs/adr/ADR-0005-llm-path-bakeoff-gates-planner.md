---
id: ADR-0005
title: Planner LLM path chosen by bakeoff, not by opinion
status: Accepted
decided_on: 2026-05-25
deciders:
  - team lead
related_adrs:
  - ADR-0003
---

# ADR-0005 — Planner LLM path chosen by bakeoff, not by opinion

## Status

Accepted.

## Context

Two technically defensible paths exist for the planner LLM hot path:

1. **All-local**: SGLang serving SeaLLMs-v3-7B AWQ-INT4 on the RTX 5070. Deterministic, no network dependency at the venue.
2. **Cloud-augmented**: Groq Llama-3.3-70B (sub-100 ms TTFT, OpenAI-compatible API). 5–10× faster than OpenAI but adds a network dependency.

The original strategy doc had no measurement for this; the choice was guesswork. The May 25 meeting concluded we should let a benchmark decide. Groq does not serve VLMs, so this bakeoff is for the planner only — the VLM reranker remains local Vintern-3B-beta (with optional Gemini 2.5 Flash escalation, separate decision).

## Decision

The choice of online planner-LLM path is decided by the bakeoff in [`docs/proposals/09-llm-path-bakeoff.md`](../proposals/09-llm-path-bakeoff.md). The bakeoff:

- Tests four paths (A all-local, B1 hybrid Groq-70B, B2 hybrid Groq-8B, C all-cloud).
- Runs against a 300-task mock set spanning KIS / QA / Ad-hoc.
- Pre-registers a 5-criterion gate: **p95 < 2 s · failure < 0.5 % · valid_json ? 99 % · R@10 within 1 % of best · cost < $1/round**.
- Tests three network conditions (Vietnamese 5G primary, 4G hotspot, netem-throttled) at three times of day to catch Groq peak-load variance.
- Freezes the criteria at the commit SHA of the bakeoff proposal; criteria cannot be re-specified after the first run.

**Deadline**: end of June 2026, before Phase 2 fine-tuning starts.

**Owner**: the team lead (named in proposal 09 §12).

Until the bakeoff completes, code that depends on the planner path should be parametric (config flag selects local-SGLang vs cloud-Groq client) to avoid blocking other work.

## Consequences

### Positive
- Closes the "cloud budget" question with a measurement, not a guess.
- The frozen criteria + raw-metric commit policy in proposal 09 §13 prevent post-hoc rationalisation.
- Other Phase 1 work can proceed against the parametric planner interface.

### Negative
- Phase 2 fine-tuning cannot finalise its target model until the bakeoff completes. We mitigate by training C1/C2/C4 against the *interface*, not the specific planner model.
- One person (the owner) runs the benchmark; team must accept that result barring procedural violation.

### Neutral / observable
- Proposals 01 §5.8 and 02 §4 will be updated post-bakeoff to reflect the chosen path.

## Alternatives considered

- **Pick local now and don't measure** — fastest in Phase 1 — rejected because we have no evidence Groq isn't faster end-to-end on Vietnamese 5G, and gambling on the venue network is exactly what last year's team learnt to avoid.
- **Pick Groq now and don't measure** — same logic, opposite bet — rejected for the same reason.
- **Run the bakeoff after Phase 1** — defer the decision — rejected because Phase 2 (fine-tuning, original contributions) needs to target a concrete planner model.

## References

- [`docs/proposals/09-llm-path-bakeoff.md`](../proposals/09-llm-path-bakeoff.md)
- [`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) §5.8
- Groq API docs — <https://console.groq.com/docs>
- SGLang grammar-constrained decoding — <https://docs.sglang.ai/sampling_params.html>
