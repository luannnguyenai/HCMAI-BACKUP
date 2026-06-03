---
id: SPEC-0025
title: Encoder bake-off (Qwen3-VL-Embedding vs SigLIP-2 / Meta CLIP 2 / provided CLIP)
status: Implementing
owner: unassigned
created: 2026-06-02
updated: 2026-06-02
implements_proposal: docs/proposals/01-interactive-system-architecture.md SS 5.3 (encoder selection)
related_adrs:
  - ADR-0003
  - ADR-0006
  - ADR-0007
depends_on:
  - SPEC-0004
---

# SPEC-0025 - Encoder bake-off: Qwen3-VL-Embedding vs the floor

> A small offline bake-off that screens **Qwen3-VL-Embedding-2B** against our [SPEC-0004](SPEC-0004-image-embedding-service.md) floor (SigLIP-2, Meta CLIP 2) and the organisers' provided **CLIP ViT-B/32** baseline on the real AIC2025 proxy corpus ([research-note 07](../research-notes/07-aic2025-proxy-corpus.md)), to inform the encoder-selection question raised in the 2026-06-02 team meeting. This is a **directional screen**, not the final accuracy verdict: rigorous R@k is gated on ground truth (SS 9 Q1).

## 1. Context

The 2026-06-02 meeting asked whether to adopt Gemini Embedding 2 or Qwen3-VL-Embedding for the offline media pipeline. The encoder-selection report concluded Gemini is **disqualified** for the retrieval path (closed-weight API; the query encoder must run on the air-gapped 5070 per [ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md)), while Qwen3-VL-Embedding (Apache-2.0; 2B + 8B; MMEB-V2 SOTA, strong on visual-document retrieval) is a legitimate candidate. The real question: does the **2B** beat the floor on Vietnamese cross-modal retrieval **and** fit the 12 GB 5070 online budget (~3 GB headroom after planner + reranker)?

[Research-note 07](../research-notes/07-aic2025-proxy-corpus.md) makes this **measurable now** rather than June-25-blocked: the AIC2025 proxy corpus on the box has 121,457 keyframes (1280x720, L25-L30) plus the official Vietnamese KIS query set. Two GT-free axes are useful directional screens:

- **Deployability** (the ADR-0003 gate): per-encoder query-path peak VRAM (FP16 + INT4) and indicative H200 query latency. VRAM is **hardware-independent** - a portable proxy for the 5070 fit; absolute 5070 latency is a follow-up needing the finals hardware.
- **Qualitative retrieval** (visual side-by-side): ~20 real Vietnamese KIS queries -> top-k keyframes per encoder, rendered as an HTML contact sheet for human judgment - especially on text-heavy / visual-document keyframes (Qwen's MMEB strength).

Rigorous **R@k / nDCG** is out of scope until either the 2025 baseline author (`ThanhToan2111`, on the team) shares his DRES answer keys ([research-note 05](../research-notes/05-baseline-2025-analysis.md) Q1), or the June-25 judged set lands.

## 2. Scope

### 2.1 In scope

- **`MetaClip2Embedder`** - Meta CLIP 2 ViT-H/14 (`open_clip` `ViT-H-14-quickgelu` / `metaclip-2-worldwide`; 1024-d). Mirrors `SigLip2Embedder`'s lazy-torch + `embedding`-extra pattern; `encode_image` + `encode_text`.
- **`Qwen3VLEmbedder`** - Qwen3-VL-Embedding-2B (`transformers`, `trust_remote_code=True` per the HF card; 2048-d native, optional MRL truncation via `out_dim`). Unified `encode_image` + `encode_text`. Lazy-torch + `embedding`-extra.
- **`ProvidedClipEmbedder`** - the organisers' weak baseline. `encode_image` **loads** pre-extracted `clip-features-32-aic25-b1` vectors by `frame_id` (no model run); `encode_text` runs the `openai/clip-vit-base-patch32` text tower (512-d) for queries. Best-effort: only the frames whose provided features align by `frame_id` are scored; if alignment is unclean the lane is dropped (logged), leaving the 3 real encoders.
- **`aic2026.eval.encoder_bench`** - the bench harness: index a keyframe sample per encoder (reuse `extract.py`), run `DenseRetriever` top-k, build an **HTML contact sheet** (per query: Vietnamese text + one row of top-k thumbnails per encoder), and `measure_deployability(encoder, sample_texts)` -> peak VRAM (FP16, and INT4 where the encoder supports it) + query latency p50/p95.
- **`bin/embed bench`** subcommand: `--kf-root`, `--queries`, `--encoders`, `--n-docs` (sample size), `--top-k`, `--out`. Plus `infra/remote/encoder_bench.sh` to drive it on the box.
- **Deliverables**: a deployability table (markdown/JSON) and `bench_report.html`, both written into / linked from this spec's SS 10 changelog after the lease run.

### 2.2 Out of scope

- **R@k / nDCG accuracy** - needs ground truth (SS 1). This spec is deployability + qualitative only.
- **InternVideo2** and any video encoder (the bench is image-text).
- **Milvus / ANN** - the bench uses brute-force `DenseRetriever` over a bounded sample; production indexing is SPEC-0006.
- **Actual INT4 quantisation kernels** on the 5070 (ADR-0006) - we measure the INT4 *footprint* (e.g. via `bitsandbytes` load or a documented size estimate), not a finals deployment.
- **Fusion** of any adopted encoder into the ranker ensemble - that is SPEC-0015 (C2).
- **The full 121,457-frame index** - the bench samples ~20k; full-corpus indexing is a post-decision follow-up.

## 3. API contract / interface

```python
# aic2026/embedding/qwen3vl_embed.py  (mirrors siglip2.py)
class Qwen3VLEmbedder:
    model_id: str = "qwen3-vl-embedding-2b"
    dim: int  # 2048 native, or the MRL out_dim if set
    def __init__(self, *, device: str = "cpu", dtype: str = "float16",
                 out_dim: int | None = None, load_in_4bit: bool = False) -> None: ...
    def encode_text(self, texts: list[str]) -> np.ndarray: ...   # (n, dim) L2-normalised
    def encode_image(self, paths: list[Path]) -> np.ndarray: ...  # (n, dim) L2-normalised

# aic2026/embedding/metaclip2.py    -> MetaClip2Embedder  (1024-d, open_clip)
# aic2026/embedding/provided_clip.py -> ProvidedClipEmbedder(features_dir, ...) (512-d)
```

```python
# aic2026/eval/encoder_bench.py
@dataclass(frozen=True)
class DeployStat:
    model_id: str; dim: int; params_m: float | None
    vram_fp16_mb: float | None; vram_int4_mb: float | None
    latency_p50_ms: float; latency_p95_ms: float
    fits_5070_headroom: bool   # INT4 footprint <= headroom_mb

def measure_deployability(encoder, sample_texts: list[str], *, headroom_mb: float = 3072) -> DeployStat: ...

def run_qualitative(
    encoders: dict[str, Embedder], query_texts: list[str], doc_paths: list[Path],
    *, top_k: int = 5, out_html: Path,
) -> None: ...   # encodes docs+queries per encoder, DenseRetriever top-k, writes the contact sheet
```

```
bin/embed bench --kf-root DIR --queries FILE_OR_DIR
                --encoders dummy,siglip2,metaclip2,qwen3vl,provided
                [--n-docs 20000] [--top-k 5] [--out DIR]
```

## 4. Behaviour

- **Encoders** follow the SPEC-0004 `Embedder` protocol: `float32 (n, dim)`, rows L2-normalised; heavy deps lazy-imported; missing deps raise the `embedding`-extra hint. `ProvidedClipEmbedder.encode_image` reads cached vectors (raises if a `frame_id` is absent unless `strict=False`, which zero-fills + logs).
- **Sampling** is deterministic given `(kf_root, n_docs, seed)`. Each encoder indexes the **same** sampled paths so the side-by-side is apples-to-apples.
- **Qualitative report**: `bench_report.html` has one section per query (the Vietnamese text) and, under it, one row per encoder showing the top-`k` thumbnails (each captioned with `frame_id` + score). Thumbnails are `<img>` tags pointing at the on-box frame paths (the HTML is viewed where the frames live, or with a `--copy-thumbs` option that copies them next to the HTML).
- **Deployability**: `measure_deployability` times `encode_text` over `sample_texts` (warmup + repeated runs -> p50/p95) and records `torch.cuda.max_memory_allocated` deltas around model load for FP16 and, when `load_in_4bit`, INT4. On CPU (CI), VRAM fields are `None` and only latency + shape are exercised.
- **Determinism / CI**: the harness logic (sampling, top-k panel structure, deploy-stat shape) is exercised with `DummyEmbedder` on CPU; the real encoders are torch-gated and skipped in CI.

## 5. Acceptance criteria

- **AC1**: each new encoder (`MetaClip2Embedder`, `Qwen3VLEmbedder`, `ProvidedClipEmbedder`) satisfies the `Embedder` protocol - `encode_text`/`encode_image` return `float32 (n, dim)` unit-norm rows of the declared `dim` - and lazy-imports torch (no `import torch` at module import time). Verified in `tests/unit/test_encoders_AC1.py` (`importorskip`-gated for the live-deps path; the lazy-import contract is checked in a fresh subprocess like `test_embedding_siglip2_AC4.py`).
- **AC2**: `ProvidedClipEmbedder.encode_image` returns the pre-extracted vector for a known `frame_id` and obeys `strict`/zero-fill on a missing id. Verified in `tests/unit/test_provided_clip_AC2.py` with a tiny fixture feature store (no network).
- **AC3**: `run_qualitative` over `DummyEmbedder` encoders produces an HTML file containing one section per query and one row per encoder with exactly `top_k` thumbnail entries, gold-agnostic (no GT). Verified in `tests/unit/test_encoder_bench_AC3.py`.
- **AC4**: `measure_deployability` returns a `DeployStat` whose latency percentiles are positive, `dim` matches the encoder, and `fits_5070_headroom` is the boolean `int4_footprint <= headroom_mb` (CPU path: VRAM `None`, latency measured). Verified in `tests/unit/test_encoder_bench_AC4.py` (DummyEmbedder).
- **AC5**: spec + module docstrings document the directional-screen framing (no GT), the deployability-on-H200-vs-5070 caveat, and the `embedding`-extra packaging. Verified by inspection.

## 6. Non-functional requirements

- **CI**: CPU-only, no torch, no network - DummyEmbedder exercises the harness; real encoders skipped.
- **Compute** (lease): index ~20k frames x 4 encoders on the H200 ~1-1.5 h; retrieval + deployability + HTML ~minutes. Bounded well within the lease.
- **Determinism**: SHA-seeded sampling; `DenseRetriever` is pure numpy.
- **Compatibility**: Python 3.11+; `embedding` extra: existing `torch`/`open-clip-torch`/`transformers`/`pillow` + a `transformers` floor that supports Qwen3-VL-Embedding (pin in SS 7) + optional `bitsandbytes` for the INT4 footprint measurement.

## 7. Dependencies

- **Internal**: SPEC-0004 (`Embedder`, `extract.py`), `aic2026.eval.retrievers.DenseRetriever`, `aic2026.eval.demo` formatting precedent.
- **External** (added to the `embedding` extra): `bitsandbytes` (optional, INT4 footprint); a `transformers` version supporting `Qwen3-VL-Embedding` (verify on the box; bump the floor if the installed one is too old). Meta CLIP 2 + the provided-CLIP text tower load via the already-present `open_clip` / `transformers`.
- **Data**: AIC2025 proxy corpus on the box (keyframes + KIS query texts + `clip-features-32-aic25-b1`); permission caveat per research-note 07 SS 5.

## 8. Test plan

- **Unit (CPU, offline)**: `test_encoders_AC1.py` (lazy-import + protocol, importorskip live), `test_provided_clip_AC2.py` (fixture feature store), `test_encoder_bench_AC3.py` (HTML structure via DummyEmbedder), `test_encoder_bench_AC4.py` (DeployStat shape/latency).
- **Lease run** (records evidence for the decision gate): `bash infra/remote/encoder_bench.sh` -> deployability table + `bench_report.html` over ~20 KIS queries x ~20k frames x 4 encoders.

## 9. Open questions

- **Q1 (blocks R@k)**: no 2025 ground truth in the proxy drop (research-note 07). Rigorous accuracy needs `ThanhToan2111`'s DRES keys or June-25. Until then the bench is deployability + qualitative.
- **Q2 (RESOLVED on box 2026-06-02)**: Qwen3-VL-Embedding-2B is **not** a plain `transformers.AutoModel` (that loads only the base `Qwen3VLModel`, no embedding head -> our first mean-pool attempt errored). The official API is the QwenLM/Qwen3-VL-Embedding repo's own `Qwen3VLEmbedder` class (`src/models/qwen3_vl_embedding.py`) with instruction-aware list-of-dict inputs + a `.process()` method. `Qwen3VLEmbedder` now **delegates** to it (`impl_src` = cloned repo, + `qwen-vl-utils`); transformers 5.9 works. Verified: `text(2,2048) image(2,2048)`, unit-norm.
- **Q3 (RESOLVED on box)**: provided `clip-features-32` is **per-video matrices** (`L25_V011.npy` -> `(318, 512)` float16), not per-frame. `ProvidedClipEmbedder` gained a per-video loader keyed `<video>_<NNN>` aligning row `i` to keyframe `<video>/<i+1:03d>.jpg`. (Assumption: organiser keyframes are 1-based 3-digit + row-aligned; holds for the sampled frames.)
- **Q4**: INT4 footprint - `bitsandbytes` is wired but the bench measured FP16 only this pass; see SS 10 for the per-encoder-VRAM measurement caveat + the param-based INT4 estimate.
- **Q5 (new)**: the deployability VRAM was measured with all 4 models resident (cumulative `max_memory_allocated`), so it does **not** isolate per-encoder footprint. Latency is per-encoder and valid. Isolated per-encoder VRAM (load one model at a time; and for CLIP encoders, the **text tower only** since that is what runs online) is a harness follow-up.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-06-02 | implementer (user-directed) | Created at status **Implementing** (solo-flow, plan-approved). Scopes the Qwen3-VL-Embedding vs SigLIP-2 / Meta CLIP 2 / provided-CLIP bake-off on the AIC2025 proxy corpus: deployability (5070-fit) + qualitative side-by-side; rigorous R@k deferred (no GT). Implements 3 new `Embedder`s + `eval/encoder_bench` + `bin/embed bench` + a remote runner. Lease results to be appended here. |
| 2026-06-03 | implementer | **Lease run on the H200 (8,000 keyframes x 4 encoders, GPU 1; GPU 0 was held by a stale 132 GB process).** All four encoders validated on real data after on-box integration fixes (Q2/Q3): siglip2 (1152-d), metaclip2 (1024-d), qwen3vl (2048-d via the official `.process()`), provided CLIP (512-d). **Deployability - query-encode latency p50 (H200, FP16):** siglip2 **10.6 ms**, metaclip2 **12.3 ms**, provided 11.0 ms, **qwen3vl 52.7 ms (~5x the floor)**. VRAM figures in `deployability.json` are cumulative (all 4 models resident; Q5) - not per-encoder - so the `fits_5070_headroom=false` flags are an artifact, not the real verdict. **Architectural footprint read (the real 5070 gate):** the CLIP-style floor hosts only a lightweight *text tower* online (~150-600 MB), whereas Qwen is a *unified 2B* with no separate light text tower, so its online path is the full model (~4.5 GB FP16; ~1.5-2 GB INT4 by param estimate - feasible in the ~3 GB headroom but tight, and it is the dominant online cost). **Qualitative:** `bench_report.html` (top-5 per query x 4 encoders over a 8k-frame sample) generated for human review; not self-judged here (no GT; subset means the exact answer frame is often absent, so it screens *topical* relevance, not recall). **Decision (directional):** do **not** adopt Qwen-2B as the online query encoder now - ~5x latency + full-2B online footprint + no accuracy edge demonstrated (R@k blocked on GT). Its likely real value is an **offline visual-document lane** (8B, fused), to be confirmed once GT exists. Floor (SigLIP-2 + Meta CLIP 2) remains the online encoder set. |
