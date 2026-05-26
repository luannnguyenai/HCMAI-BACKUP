---
id: ADR-0006
title: INT4 (AWQ) / FP4 quantization for all hot-path LLM and VLM weights
status: Accepted
decided_on: 2026-05-25
deciders:
  - team lead
related_adrs:
  - ADR-0003
---

# ADR-0006 — INT4 (AWQ) / FP4 quantization for all hot-path LLM and VLM weights

## Status

Accepted.

## Context

ADR-0003 commits us to a 12 GB RTX 5070 for the inference path at finals. The hot-path models in their default precisions do not fit:

| Model | FP16 size | Required action |
|---|---|---|
| SeaLLMs-v3-7B | ~14 GB | quantize |
| Vintern-3B-beta | ~7 GB | quantize |
| Qwen2.5-VL-7B (if used) | ~13 GB | quantize or replace |

Quantization options available in 2026:
- **AWQ-INT4** — Activation-aware Weight Quantization; 4-bit weights with per-channel scaling; ~99 % of full-precision accuracy on most tasks; well-supported in vLLM and SGLang.
- **GPTQ-INT4** — older; slightly worse accuracy.
- **GGUF Q4_K_M** — llama.cpp format; CPU + GPU; flexible.
- **NF4** — QLoRA-style; designed for normal-distributed weights.
- **FP4 (Blackwell native)** — RTX 5070 has tensor cores for FP4; ~2× throughput vs INT4 at similar quality.
- **MXFP4** — micro-scaled FP4; emerging.

## Decision

All hot-path LLM and VLM weights deployed to the RTX 5070 finals box are quantized:

- **Primary**: **AWQ-INT4** via vLLM or SGLang. Mature, well-supported, ~99 % accuracy retention.
- **Upgrade path** (Phase 4 if time): **FP4** via TensorRT-LLM, exploiting Blackwell's native FP4 tensor cores for ~2× throughput.
- **Calibration**: performed on the **GH200** during indexing; the calibrated quantized weights are then frozen and shipped to the 5070.
- **Quality gate**: each quantized model must pass a regression check on the 300-task dev set — R@1 within **1.0 %** of the full-precision baseline, valid_json_rate ? 99 % for the planner. If quantization regresses beyond this, ship full-precision via cloud (Groq for text, Gemini for vision) instead.
- **Emergency fallback**: pre-stage **llama.cpp + GGUF Q4_K_M** weights on the same machine. If the GPU dies mid-round we fall back to ~5–10 tok/s CPU inference. Slow, but better than forfeiting.

Models that cannot reach the quality gate at INT4 must be down-sized rather than de-quantized — e.g. drop Vintern-3B-beta ? Vintern-1B-v3.5 INT4, drop SeaLLMs-v3-7B ? Qwen2.5-3B-Instruct INT4.

## Consequences

### Positive
- The full hot path (planner + reranker + text-tower encoders + KV cache) fits in ~9 GB, leaving ~3 GB of headroom on the 12 GB 5070.
- Calibration on the GH200 is fast and one-off; runtime cost is just inference.
- FP4 upgrade path is free if we have time in Phase 4.

### Negative
- Quantization adds an extra ML-Ops step (calibration ? smoke benchmark ? ship) that must be re-run when the underlying model is updated.
- Edge cases in JSON output may fail at INT4 that pass at FP16 — the dev-set quality gate catches this but adds CI complexity.
- Some open-source quantized model checkpoints have not been audited; we should prefer to quantize ourselves on the GH200 from known-good FP16 weights.

### Neutral / observable
- Phase 2 fine-tunes (LoRA on SigLIP-2; C1 DiacriticBERT; C2 LightGBM) train against full-precision backbones on the GH200; quantization happens after fine-tuning.
- The eval harness ([`docs/proposals/05-evaluation-harness.md`](../proposals/05-evaluation-harness.md)) gains a quantization-quality slice as a mandatory pre-deploy check.

## Alternatives considered

- **FP16 with model swapping** — keep full-precision and load/unload models as needed — rejected because swap latency (multi-second from disk) is incompatible with our <100 ms hot-path SLO.
- **Cloud-only inference** — let Groq / Gemini do all the work — under bakeoff in ADR-0005; not chosen here.
- **Use smaller models everywhere** — e.g. Qwen2.5-3B as the planner — kept as a fallback path if INT4 of the 7B planner fails the quality gate.

## References

- vLLM AWQ docs — <https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html>
- SGLang quantization — <https://docs.sglang.ai/start/install.html#optional-quantization>
- TensorRT-LLM FP4 — <https://nvidia.github.io/TensorRT-LLM/quantization/fp4.html>
- [`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) §5.8 and §6 (to be updated)
