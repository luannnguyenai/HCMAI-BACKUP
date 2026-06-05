---
id: ADR-0012
title: Qwen3-VL-Embedding-2B is an offline-only visual-document lane, never the online query encoder
status: Accepted
decided_on: 2026-06-04
deciders:
  - team lead
related_adrs:
  - ADR-0003
  - ADR-0006
  - ADR-0007
  - ADR-0008
---

# ADR-0012 - Qwen3-VL-Embedding-2B is an offline-only visual-document lane, never the online query encoder

## Status

Accepted.

## Context

The 2026-06-02 team meeting asked whether to adopt Qwen3-VL-Embedding (Apache-2.0; MMEB-V2 SOTA, strong on visual-document retrieval) as a retrieval encoder, and at which size. [SPEC-0025](../specs/SPEC-0025-encoder-bench.md) ran a directional screen of the 2B against our floor (SigLIP-2, Meta CLIP 2) and the organisers' provided CLIP on the real AIC2025 proxy corpus. The lease run on 2026-06-03 produced a clear deployability signal:

- **Latency**: Qwen3-VL-Embedding-2B query-encode p50 was **52.7 ms on H200 (FP16), about 5x the floor** (SigLIP-2 10.6 ms, Meta CLIP 2 12.3 ms, provided CLIP 11.0 ms).
- **Online footprint**: the CLIP-style floor hosts only a lightweight *text tower* online (~150-600 MB), whereas Qwen is a *unified 2B* with no separate light text tower, so its online path is the full model (~4.5 GB FP16; ~1.5-2 GB INT4 by param estimate - feasible in the 5070's ~3 GB headroom but tight, and it would be the dominant online cost).
- **Accuracy**: rigorous R@k / nDCG is **blocked on ground truth** (SPEC-0025 SS 9 Q1: no 2025 DRES answer keys in the proxy drop, June-25 judged set not yet landed), so no online accuracy edge has been demonstrated.

The online hot path runs on the air-gapped 12 GB RTX 5070 at finals ([ADR-0003](ADR-0003-rtx5070-finals-gh200-offline.md)); the query encoder must run locally there. Heavy image-tower extraction already runs offline on the GH200-class box per the same ADR. The question this decision settles: do we put Qwen on the online query path at all, and if not, does it have a role?

## Decision

Qwen3-VL-Embedding-2B runs **offline only**. It produces a pre-indexed **visual-document dense lane** (its `encode_image` over keyframes), written to `.npy` + manifest by the SPEC-0004 extraction path and fused via C2 (SPEC-0015; the reserved fusion spec). It is **never** the online query encoder: online query encoding stays on the SigLIP-2 + Meta CLIP 2 text towers (the [SPEC-0004](../specs/SPEC-0004-image-embedding-service.md) floor). Adoption of the Qwen lane into the live fusion ensemble is **gated on ground-truth-proven lift** (SPEC-0025 SS 9 Q1) - until GT exists the lane is built and indexed but its fusion weight stays out of the shipped configuration. This lane is additive: the floor-only system remains a complete fallback.

## Consequences

### Positive

- Captures Qwen's strong visual-document embeddings (its MMEB strength) with **zero online latency cost** - all Qwen compute is offline, so the ~5x latency never touches the hot path.
- Keeps the online 5070 budget clean: only the lightweight CLIP-style text towers run online, preserving headroom for the planner + reranker (ADR-0006).
- Reuses the existing SPEC-0004 extraction CLI + manifest contract; no new online serving surface.

### Negative

- Extra **offline compute** (an additional encoder pass over the full keyframe corpus on the lease).
- An extra Milvus dense field for the Qwen vectors (SPEC-0006, reserved) and an extra fusion lane to train and gate (SPEC-0015, reserved).
- Closes off "Qwen as the single online unified encoder" - we explicitly do not get the one-model-for-query-and-doc simplification.

### Neutral / observable

- The Qwen lane's value is **unproven on accuracy** until ground truth lands; the build-and-index work proceeds but the fusion weight is gated, so the decision is reversible at low cost (drop the lane, keep the floor).
- Adds `qwen3vl` to the `bin/embed images` offline-extraction surface, threading `--impl-src` (cloned QwenLM/Qwen3-VL-Embedding repo) and `--out-dim` (MRL truncation). The `encode_text` method exists but is documented as off the online query path.

## Alternatives considered

- **Qwen as the single online unified encoder** - one model maps query text and keyframe images into one space, simplifying the stack - **rejected because** its query-encode latency is ~5x the floor (52.7 ms vs ~11 ms p50 on H200) and, as a unified 2B with no light text tower, its online footprint is the full model on the 12 GB 5070 (ADR-0003), with no demonstrated accuracy edge (R@k GT-blocked).
- **Gemini Embedding 2** - a strong closed-API multimodal embedder considered in the same meeting - **rejected because** it is a closed-weight API and cannot run on the air-gapped 5070 query path (ADR-0003).
- **Floor-only, no Qwen at all** - keep just SigLIP-2 + Meta CLIP 2 - **kept as the fallback** rather than chosen: the Qwen offline lane is purely additive and GT-gated, so if it shows no lift we simply do not ship its fusion weight and the floor stands alone.

## References

- [`SPEC-0025`](../specs/SPEC-0025-encoder-bench.md) - the bake-off that produced the directional screen (SS 10 changelog 2026-06-03).
- [`SPEC-0004`](../specs/SPEC-0004-image-embedding-service.md) - the embedding service whose offline extraction path hosts the Qwen lane.
- [`ADR-0003`](ADR-0003-rtx5070-finals-gh200-offline.md) - the RTX 5070 / GH200 hardware split that makes "offline-only" meaningful.
- [`ADR-0006`](ADR-0006-int4-quantization-hot-path.md) - INT4 / FP4 on the hot path; the online budget Qwen would have competed for.
- [`ADR-0007`](ADR-0007-original-contributions-c1-c2-c4.md) - C1/C2/C4; the Qwen lane is a C2 fusion input.
- [`ADR-0008`](ADR-0008-rrf-as-runtime-fallback.md) - C2 is the default fusion, RRF the fallback; the lane fuses via C2.
