---
id: ADR-0003
title: RTX 5070 at finals; GH200 cloud burst for offline training and indexing
status: Accepted
decided_on: 2026-05-25
deciders:
  - team lead
related_adrs:
  - ADR-0006
---

# ADR-0003 — RTX 5070 at finals; GH200 cloud burst for offline training and indexing

## Status

Accepted.

## Context

Initial proposals assumed an NVIDIA A6000 (48 GB) as the inference target at finals. In the May 25 meeting the team confirmed that the actual finals hardware is an **RTX 5070 (12 GB VRAM, Blackwell)** — consumer-class, no NVLink, but with native FP4 tensor cores. Offline, the team has access to **GH200** (96 GB HBM3e) cloud burst for training and indexing work.

This invalidates the original "all models loaded online" sizing in [`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) §6. Naive hosting (SigLIP-2 FP16 + Meta CLIP 2 FP16 + SeaLLMs-v3-7B + Vintern-3B fp16) would require ~47 GB — almost 4× the 5070's 12 GB.

## Decision

Online (RTX 5070) hosts only:
1. **Text-tower encoders** of the dual encoders (SigLIP-2 text ~150 MB INT8, Meta CLIP 2 XLM-R-large text ~560 MB INT8, BGE-M3 text ~600 MB INT8). Image embeddings are pre-indexed offline; image towers never run online.
2. **Planner LLM** at INT4 (SeaLLMs-v3-7B AWQ-INT4 ~4.5 GB or FP4 ~4.0 GB).
3. **VLM-as-judge reranker** at INT4 (Vintern-3B-beta ~2 GB).
4. KV cache + activations + Milvus client buffers (~2 GB headroom).

Total online VRAM: **~9 GB**, leaving ~3 GB of headroom.

Offline (GH200 cloud burst) handles:
- All image-tower embedding extraction (SigLIP-2, Meta CLIP 2, InternVideo2).
- PhoWhisper-large + WhisperX on the full audio corpus.
- PaddleOCR + VietOCR on all keyframes.
- Qwen2.5-VL-72B INT4 Vietnamese captioning.
- DiacriticBERT (C1) training and DreamLIP-style synthetic-caption fine-tuning.
- Quantization calibration of the hot-path models before they ship to the 5070.

Finals deployment: two identical 5070-equipped laptops (mirror images), pre-staged weights, no network dependency for the inference path.

## Consequences

### Positive
- Forces an architecture that is cheaper, faster, and reproducible by other teams on commodity hardware.
- Removes A6000 / 4090 as a single point of procurement failure.
- Native FP4 on Blackwell gives us a quantization upgrade path (post-AWQ-INT4) at zero additional cost.

### Negative
- 12 GB leaves no headroom for the larger 7B–13B image-image embedding (e.g. EVA-CLIP-E). C5 counterfactual rerank is now harder to ship locally.
- Loss of A6000's 48 GB headroom means we cannot host a 72B reranker locally; cloud Gemini Flash is the only path for that.

### Neutral / observable
- All hot-path models must be quantized — ADR-0006 captures this.
- Image-similarity (image-as-query) at runtime needs the image tower; we either load it on-demand (evicting the planner) or restrict the feature.
- Proposal 01 §6 hardware sizing must be rewritten before Phase 1.

## Alternatives considered

- **Procure an A6000 or 4090 for finals** — restores original sizing — rejected because procurement is not feasible in 17 weeks within budget.
- **Run the planner/reranker entirely on cloud** — frees all 12 GB — under bakeoff (see ADR-0005 + [proposal 09](../proposals/09-llm-path-bakeoff.md)); decision deferred to measurement.
- **Use 2× RTX 4060 (16 GB each)** — splits the load — rejected because consumer cards have no NVLink and tensor-parallel splitting of 7B models incurs latency overhead exceeding the gain.

## References

- [`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) §6 (to be updated)
- [`docs/proposals/09-llm-path-bakeoff.md`](../proposals/09-llm-path-bakeoff.md) — bakeoff between local-5070 and Groq-cloud planner paths
- NVIDIA Blackwell FP4 — <https://www.nvidia.com/en-us/data-center/blackwell-architecture/>
