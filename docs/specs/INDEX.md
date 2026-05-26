# Spec Registry

> Append a row when you reserve a spec ID, even before the spec body is written. Keep the table sorted by ID. Status is the single source of truth — update it as the spec moves through the lifecycle in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Conventions

- ID format: `SPEC-NNNN` (4 digits, never reuse).
- File path: `docs/specs/SPEC-NNNN-kebab-name.md`.
- Status one of: `Draft` | `Review` | `Approved` | `Implementing` | `Implemented` | `Deprecated`.

## Specs

| ID | Title | Status | Owner | Proposal | Notes |
|---|---|---|---|---|---|
| [SPEC-0001](SPEC-0001-evaluation-harness.md) | Evaluation harness (mock-DRES + 300-task set) | Draft | _unassigned_ | 05 | Foundational; gate for everything else |
| [SPEC-0002](SPEC-0002-llm-path-bakeoff-runner.md) | LLM path bakeoff runner | Draft | team lead | 09 | Owns the June 25 decision |
| _SPEC-0003_ | Data ingestion pipeline | _reserved_ | _unassigned_ | 03 | TransNetV2 + keyframe + audio extraction |
| _SPEC-0004_ | Image-embedding service | _reserved_ | _unassigned_ | 01 SS 5.3 | SigLIP-2 + Meta CLIP 2 + InternVideo2 |
| _SPEC-0005_ | OCR + ASR ingestion | _reserved_ | _unassigned_ | 01 SS 5.5–5.6 | PaddleOCR + PhoWhisper |
| _SPEC-0006_ | Milvus schema and queries | _reserved_ | _unassigned_ | 01 SS 5.4 | Hybrid vector + structured filter |
| _SPEC-0007_ | Elasticsearch schema and queries | _reserved_ | _unassigned_ | 01 SS 5.4 | OCR / ASR / caption indexes |
| _SPEC-0008_ | Planner LLM service | _reserved_ | _unassigned_ | 01 SS 5.8 | SGLang + SeaLLMs-v3 or Groq; bakeoff-gated |
| _SPEC-0009_ | Tool registry contract | _reserved_ | _unassigned_ | 02 SS 4 | Pydantic schema for tools |
| _SPEC-0010_ | VLM-as-judge reranker | _reserved_ | _unassigned_ | 01 SS 5.9 | Vintern-3B-beta + position-bias mitigation |
| _SPEC-0011_ | DANTE DP for TRAKE | _reserved_ | _unassigned_ | 01 SS 5.10 | 4-scene temporal alignment |
| _SPEC-0012_ | React operator console | _reserved_ | _unassigned_ | 06 | UI shell + grid + scrubber |
| _SPEC-0013_ | Submission verification panel | _reserved_ | _unassigned_ | 06 SS 3.7 | Anti-foot-gun UI component |
| _SPEC-0014_ | C1 — DiacriticBERT training | _reserved_ | _unassigned_ | 08 SS 3 | Diacritic-noise schedule + InfoNCE |
| _SPEC-0015_ | C2 — Per-task-type learned fusion | _reserved_ | _unassigned_ | 08 SS 4 | LightGBM LambdaRank + RRF fallback |
| _SPEC-0016_ | C4 — Agent self-distillation | _reserved_ | _unassigned_ | 08 SS 6 | DSPy MIPRO over operator traces |
| _SPEC-0017_ | LangGraph automatic-track agent | _reserved_ | _unassigned_ | 02 | State machine + retry loop |
| _SPEC-0018_ | DRES integration | _reserved_ | _unassigned_ | 05 | Submission client + score polling |
| _SPEC-0019_ | Operator trace logger | _reserved_ | _unassigned_ | 02 SS 8 | Feeds C4; Parquet append-only |

Rows in italics are reserved IDs — the spec body has not yet been authored. When you start authoring a reserved spec, drop the italics, fill in the file, set status to `Draft`.
