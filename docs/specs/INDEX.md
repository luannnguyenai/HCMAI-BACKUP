# Spec Registry

> Append a row when you reserve a spec ID, even before the spec body is written. Keep the table sorted by ID. Status is the single source of truth ? update it as the spec moves through the lifecycle in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Conventions

- ID format: `SPEC-NNNN` (4 digits, never reuse).
- File path: `docs/specs/SPEC-NNNN-kebab-name.md`.
- Status one of: `Draft` | `Review` | `Approved` | `Implementing` | `Implemented` | `Deprecated`.

## Specs

| ID | Title | Status | Owner | Proposal | Notes |
|---|---|---|---|---|---|
| [SPEC-0001](SPEC-0001-evaluation-harness.md) | Evaluation harness (mock-DRES + 300-task set) | Approved | _unassigned_ | 05 | Foundational; gate for everything else. Tier 1 (AC1+AC2+AC4) implementation in flight on branch `spec/0001-tier1-stub-harness` |
| [SPEC-0002](SPEC-0002-llm-path-bakeoff-runner.md) | LLM path bakeoff runner | Draft | team lead | 09 | Owns the June 25 decision |
| _SPEC-0003_ | Data ingestion pipeline | _reserved_ | _unassigned_ | 03 | Organisers ship keyframes + OD; shot detection (TransNetV2 / AutoShot / TransVLM) demoted to Phase 2 contingency. See [research-note 06 SS 2.2](../research-notes/06-aic2026-dataset-shape.md) |
| [SPEC-0004](SPEC-0004-image-embedding-service.md) | Image-embedding service | Implementing | _unassigned_ | 01 SS 5.3 | SigLIP-2 + Meta CLIP 2 + InternVideo2. Slice in PR: `Embedder` protocol + `DummyEmbedder` + SigLIP-2 + `bin/embed` extraction CLI. Add organisers' pre-computed CLIP as a 4th baseline lane in SPEC-0006 (research-note 06 SS 2.3). Adds the Qwen3-VL-Embedding-2B **offline-only visual-document lane** ([ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md)): `bin/embed images --encoder qwen3vl` (AC6) |
| _SPEC-0005_ | OCR + ASR ingestion (yt-dlp primary, PhoWhisper fallback) | _reserved_ | _unassigned_ | 01 SS 5.5?5.6 | PaddleOCR + two-source ASR: yt-dlp YouTube captions primary, PhoWhisper fallback (research-note 06 SS 2.1). Provenance tracked via `source: yt-dlp / phowhisper` |
| [SPEC-0006](SPEC-0006-milvus-schema-and-queries.md) | Milvus schema and queries | Implementing | _unassigned_ | 01 SS 5.4 | Single multi-vector collection (siglip2 1152 / metaclip2 1024 / qwen3vl 2048 floor) keyed by frame; offline ingest from SPEC-0004 `.npy`+manifest (R2-banked, [ADR-0011](../adr/ADR-0011-r2-artifact-store-and-lease-rollover.md)); online per-field ANN (HNSW, IP=cosine) + structured filter on `youtube_url`/`description`/OD tags. qwen3vl is the offline-only doc lane ([ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md)); qwen8b 4096 + organiser CLIP 512 deferred. Fusion is SPEC-0015. `aic2026/index` + `bin/index` CLI; AC1-AC7 on Milvus Lite (dev/CI), standalone HNSW on the lease |
| _SPEC-0007_ | Elasticsearch schema and queries | _reserved_ | _unassigned_ | 01 SS 5.4 | OCR / ASR / caption / `description` indexes. **Must NOT use `asciifolding` filter on Vietnamese text ? strips diacritics** (research-note 05 SS 4.1). `description` from organiser metadata adds a 4th text lane (research-note 06 SS 2.4) |
| _SPEC-0008_ | Planner LLM service | _reserved_ | _unassigned_ | 01 SS 5.8 | SGLang + SeaLLMs-v3 or Groq; bakeoff-gated |
| _SPEC-0009_ | Tool registry contract | _reserved_ | _unassigned_ | 02 SS 4 | Pydantic schema for tools |
| _SPEC-0010_ | VLM-as-judge reranker | _reserved_ | _unassigned_ | 01 SS 5.9 | Vintern-3B-beta + position-bias mitigation |
| _SPEC-0011_ | DANTE DP for TRAKE | _reserved_ | _unassigned_ | 01 SS 5.10 | 4-scene temporal alignment |
| _SPEC-0012_ | React operator console | _reserved_ | _unassigned_ | 06 | UI shell + grid + scrubber |
| _SPEC-0013_ | Submission verification panel | _reserved_ | _unassigned_ | 06 SS 3.7 | Anti-foot-gun UI component |
| [SPEC-0014](SPEC-0014-diacritic-bert.md) | C1 ? DiacriticBERT training | Implementing | _unassigned_ | 08 SS 3 | Diacritic-noise schedule + InfoNCE; **trainable now on public Vi caption+ASR text** (no June-25 dep). Failure mode in baseline (research-note 05 SS 4.1) |
| _SPEC-0015_ | C2 ? Per-task-type learned fusion | _reserved_ | _unassigned_ | 08 SS 4 | LightGBM LambdaRank + RRF fallback. Qwen3-VL-Embedding-2B offline visual-document lane is a (GT-gated) fusion input ([ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md)) |
| _SPEC-0016_ | C4 ? Agent self-distillation | _reserved_ | _unassigned_ | 08 SS 6 | DSPy MIPRO over operator traces |
| _SPEC-0017_ | LangGraph automatic-track agent | _reserved_ | _unassigned_ | 02 | State machine + retry loop |
| [SPEC-0018](SPEC-0018-dres-integration.md) | DRES integration client | Draft | _unassigned_ | 05 | Login + submit; borrows from 2025 baseline under [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md). Prod URL: `https://eventretrieval.oj.io.vn` |
| _SPEC-0019_ | Operator trace logger | _reserved_ | _unassigned_ | 02 SS 8 | Feeds C4; Parquet append-only |
| [SPEC-0020](SPEC-0020-ndcg-at-10-metric.md) | NDCG@10 ranking metric in the eval harness | Implemented | _unassigned_ | 05 SS 14 | C2 ship-gate metric ([ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md), [ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md)); extends SPEC-0001 harness. Merged in PR #5 |
| [SPEC-0021](SPEC-0021-ci-pipeline.md) | CI pipeline (lint + test + smoke-eval gate) | Implemented | _unassigned_ | 05 SS 6 | GitHub Actions; enforces per-PR smoke ([ADR-0009](../adr/ADR-0009-sdd-workflow.md)). Score-threshold gating deferred to SPEC-0001 AC7. Merged in PR #6 |
| [SPEC-0022](SPEC-0022-remote-gpu-runner.md) | Remote GPU job runner + Cloudflare R2 artifact sync | Implementing | _unassigned_ | 05 SS 5 | `bin/remote` CLI. R2 ([ADR-0011](../adr/ADR-0011-r2-artifact-store-and-lease-rollover.md)) for cross-lease persistence. First job: `extract-siglip` wraps SPEC-0004. cache-weights added in spec/0023 |
| [SPEC-0024](SPEC-0024-provision-packaging.md) | One-command provisioning + R2 warm-cache restore | Implementing | _unassigned_ | 05 SS 5 | Hardens SPEC-0022 `provision`; `cache-env` job; fixes `R2Client.list()` R2-checksum bug + run_id trap from the H200 lease |
| [SPEC-0025](SPEC-0025-encoder-bench.md) | Encoder bake-off (Qwen3-VL-Embedding vs floor) | Implemented | _unassigned_ | 01 SS 5.3 | Directional screen on the AIC2025 proxy: deployability (5070-fit) + qualitative side-by-side. Extends SPEC-0004 `Embedder`; new MetaClip2/Qwen3VL/ProvidedClip + `eval/encoder_bench` + `bin/embed bench`. Rigorous R@k deferred (no GT). decision: [ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md) (Qwen offline-only) |
| [SPEC-0026](SPEC-0026-mvp-serving-api.md) | MVP serving API (Vietnamese KIS query + image serving + issue capture) | Implemented | _unassigned_ | 01 SS 5.1 | FastAPI + WebSocket wrapper over the merged SPEC-0006 `MilvusBackend`. KIS-only; single-lane or RRF ([ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md)); QA/TRAKE/agent/C2 deferred. depends_on SPEC-0001, SPEC-0004, SPEC-0006. Decisions: [ADR-0013](../adr/ADR-0013-mvp-single-shared-server-from-r2.md) (shared server from R2), [ADR-0014](../adr/ADR-0014-mvp-reuse-milvus-backend-standalone.md) (reuse MilvusBackend on standalone), [ADR-0015](../adr/ADR-0015-keyframe-image-hosting-thumbnails.md) (thumbnail tier); UI stack [ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md) |
| [SPEC-0027](SPEC-0027-mvp-operator-ui.md) | MVP operator UI (React + WebSocket KIS console) | Implemented | _unassigned_ | 06 | proposal-06 subset: query box + virtualised thumbnail grid + frame detail + lane selector + in-UI issue capture. depends_on SPEC-0026. Stack mandate [ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md); thumbnails [ADR-0015](../adr/ADR-0015-keyframe-image-hosting-thumbnails.md). TRAKE/QA/planner/verification-bar out of scope |
| [SPEC-0028](SPEC-0028-r2-preflight-check.md) | R2 prefix precondition check for lease jobs | Implementing | _unassigned_ | 05 SS 5 | Executable form of [ADR-0016](../adr/ADR-0016-data-durability-three-tier-banking.md) rule (c): `check_prefixes`/`require_prefixes` over `R2Client` + `bin/remote preflight` + opt-in `run --require-prefix` guard. Read-only; existing flows unchanged. depends_on [SPEC-0022](SPEC-0022-remote-gpu-runner.md) |

## Cross-cutting prior art

[`docs/research-notes/05-baseline-2025-analysis.md`](../research-notes/05-baseline-2025-analysis.md) catalogues reusable patterns from the 2025 baseline. When you author any of the reserved specs above, check that note for prior art and borrow under [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md). Specifically:

- **SPEC-0003** (data ingestion): borrow `transnetv2_pytorch.py` + the 29 MB weight directly.
- **SPEC-0007** (Elasticsearch): do **not** reuse their `asciifolding` analyser; use ICU + pyvi instead.
- **SPEC-0008** (planner): reuse `auto_expand` / `num_expansions` paraphrase pattern; replace Phi-3 with SeaLLMs-v3.
- **SPEC-0009** (tool registry): reuse the request-schema knob set (`rrf_k`, `decay_rate`, `per_event_k`, etc.).
- **SPEC-0010** (VLM reranker): reuse the BLIP-2 ITC/ITM toggle as a fallback path.
- **SPEC-0011** (DANTE DP): reuse `decay_rate`, `max_gap_seconds`, `same_video_only` as parameter names.
- **SPEC-0012** (React console): port the bookmarks concept for TRAKE staging.
- **SPEC-0018** (DRES integration): already authored; borrows from the 2025 baseline directly.

## Notes

Rows in italics are reserved IDs ? the spec body has not yet been authored. When you start authoring a reserved spec, drop the italics, fill in the file, set status to `Draft`.
