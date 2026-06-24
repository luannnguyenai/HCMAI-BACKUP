# ADR Registry

> Append-only record of architectural decisions. Never reuse an ID; never edit a Accepted ADR's Decision/Context ù supersede it instead. See [`CONTRIBUTING.md`](../../CONTRIBUTING.md) for the workflow.

## Conventions

- ID format: `ADR-NNNN` (4 digits).
- File path: `docs/adr/ADR-NNNN-kebab-name.md`.
- Status one of: `Proposed` | `Accepted` | `Superseded by ADR-NNNN`.

## ADRs

| ID | Title | Status | Decided | Notes |
|---|---|---|---|---|
| [ADR-0001](ADR-0001-floor-edge-moat-strategy.md) | Strategy framed as Floor / Edge / Moat | Accepted | 2026-05-24 | Reframes the four-advantage stack |
| [ADR-0002](ADR-0002-vietnamese-capable-not-native.md) | "Vietnamese-capable" not "Vietnamese-native" | Accepted | 2026-05-24 | Honest framing of model stack |
| [ADR-0003](ADR-0003-rtx5070-finals-gh200-offline.md) | RTX 5070 at finals, GH200 for offline | Accepted | 2026-05-25 | Forces INT4 quantization on hot path |
| [ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md) | No Streamlit; React + WebSocket UI | Accepted | 2026-05-25 | Closes last year's latency root cause |
| [ADR-0005](ADR-0005-llm-path-bakeoff-gates-planner.md) | LLM path chosen by bakeoff, not opinion | Accepted | 2026-05-25 | Closes "cloud budget" open question |
| [ADR-0006](ADR-0006-int4-quantization-hot-path.md) | INT4 / FP4 quantization for hot-path models | Accepted | 2026-05-25 | Required by ADR-0003 |
| [ADR-0007](ADR-0007-original-contributions-c1-c2-c4.md) | Three primary original contributions: C1, C2, C4 | Accepted | 2026-05-24 | The "Edge" tier of ADR-0001 |
| [ADR-0008](ADR-0008-rrf-as-runtime-fallback.md) | RRF is the fallback; C2 is the default fusion | Accepted | 2026-05-24 | Inverts the original proposal stance |
| [ADR-0009](ADR-0009-sdd-workflow.md) | Spec-Driven Development is the workflow | Accepted | 2026-05-26 | The workflow itself |
| [ADR-0010](ADR-0010-borrow-from-2025-baseline.md) | Borrow from the 2025 baseline repo under attribution | Accepted | 2026-05-26 | Policy + per-file header + `THIRD_PARTY.md` |
| [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md) | Cloudflare R2 as the artifact store; 4-tier persistence across lease rollovers | Accepted | 2026-05-29 | Implemented in [SPEC-0022](../specs/SPEC-0022-remote-gpu-runner.md) |
| [ADR-0012](ADR-0012-qwen-offline-visual-document-lane.md) | Qwen3-VL-Embedding-2B is an offline-only visual-document lane, never the online query encoder | Accepted | 2026-06-04 | Screened by [SPEC-0025](../specs/SPEC-0025-encoder-bench.md); offline lane via SPEC-0004 |
| [ADR-0013](ADR-0013-mvp-single-shared-server-from-r2.md) | MVP deployment topology is one shared server fed from R2 | Accepted | 2026-06-07 | MVP test topology (not finals). Implemented by [SPEC-0026](../specs/SPEC-0026-mvp-serving-api.md); R2 source-of-truth per [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md) |
| [ADR-0014](ADR-0014-mvp-reuse-milvus-backend-standalone.md) | MVP retrieval path reuses the SPEC-0006 MilvusBackend on Milvus standalone | Accepted | 2026-06-07 | One code path, not numpy/Lite. Wrapped by [SPEC-0026](../specs/SPEC-0026-mvp-serving-api.md); FastAPI per [ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md); RRF per [ADR-0008](ADR-0008-rrf-as-runtime-fallback.md) |
| [ADR-0015](ADR-0015-keyframe-image-hosting-thumbnails.md) | Keyframe images served as a pre-generated thumbnail tier banked to R2 | Accepted | 2026-06-07 | Sourcing/hydration for ~19 GB / 121k JPGs; nginx static serving per [ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md); consumed by [SPEC-0026](../specs/SPEC-0026-mvp-serving-api.md) |
| [ADR-0016](ADR-0016-data-durability-three-tier-banking.md) | Data durability policy: three-tier banking + bank-before-consume | Accepted | 2026-06-08 | Extends [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md) after the keyframe-loss incident. Precondition check in [SPEC-0028](../specs/SPEC-0028-r2-preflight-check.md) |
