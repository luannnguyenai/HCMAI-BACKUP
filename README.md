# AIC2026 - SCOAI

> Strategy, research, and architecture for competing in the **AI Challenge HCMC 2026** (Hoi thi Thu thach Tri tue Nhan tao TP.HCM 2026), targeting **1st prize in Bang A** (university division).

**Competition site**: <https://aichallenge.hochiminhcity.gov.vn/>
**Current phase**: Phase 0 (Pre-launch alignment) - see [`docs/strategy/00-master-strategy.md`](docs/strategy/00-master-strategy.md) §1 for live dates and §10 for the open-question tracker.
**Problem**: Intelligent virtual assistant for retrieving information from large multimedia data (image + audio + text); format inherits from international **Lifelog Search Challenge (LSC)** and **Video Browser Showdown (VBS)** with a NEW **automatic-agent track** in 2026.

---

## Read this first

[`docs/strategy/00-master-strategy.md`](docs/strategy/00-master-strategy.md) - one-page summary of our plan, timeline, team, and the eight things that will lose the competition if we skip them.

[`CONTRIBUTING.md`](CONTRIBUTING.md) - the **Spec-Driven Development (SDD)** workflow every contribution follows. Read once before your first PR.

[`AGENTS.md`](AGENTS.md) - agent supplement to `CONTRIBUTING.md` for AI assistants (Cursor, Claude Code, Codex). Read once at the start of any task.

Visual companions in [`docs/illustrations/`](docs/illustrations/README.md): system architecture, UI mockup, automatic-agent loop, and the winning-hypothesis stack - use these to anchor team discussions.

![System Architecture](docs/illustrations/aic2026-system-architecture.png)

---

## What's in this repo

```
CONTRIBUTING.md                       # SDD workflow (read first)
AGENTS.md                             # agent supplement to CONTRIBUTING
pyproject.toml                        # uv-managed Python 3.11+ project
ruff.toml                             # format + lint config
.python-version                       # 3.11
uv.lock                               # pinned dependency lockfile

src/aic2026/                          # CODE
  models/                             # Pydantic shapes: task, submission, metrics
  harness/                            # backend protocol + stub, runner, scoring, aggregation (SPEC-0001)
  reporting/                          # metrics.json, report.html, README provenance
  cli/                                # Typer CLI entry point (`eval`)
  embedding/                          # Embedder protocol + SigLIP-2 / Qwen3-VL offline lane (SPEC-0004)
  index/                              # Milvus multi-vector keyframe store + queries (SPEC-0006)
  train/                              # C1 DiacriticBERT noise/corpus/train + calibrate (SPEC-0014)
  eval/                               # encoder bench + retrievers + demo (SPEC-0025)
  remote/                             # remote GPU job runner + R2 sync (SPEC-0022/0024)

tests/
  unit/                               # AC2 + AC4 + scoring + stub-backend unit tests
  integration/                        # AC1 end-to-end subprocess test
  mock_tasks/                         # 20-task placeholder smoke corpus (real Vietnamese)
    smoke_20.jsonl
    _generate_smoke.py                # authoritative generator (ASCII-safe source)

bin/
  eval                                # POSIX wrapper -> `uv run eval` (entry-point on Windows)

docs/
  strategy/
    00-master-strategy.md             # the master plan
  illustrations/                      # 4 visuals for team discussion
    aic2026-system-architecture.png
    aic2026-ui-mockup.png
    aic2026-agent-loop.png
    aic2026-winning-stack.png
    README.md                         # discussion prompts per illustration
  research-notes/                     # external research, every claim cited
    01-aic-hcmc-prior-editions.md     # 2023/24/25 lessons
    02-lsc-vbs-systems-deep-dive.md   # world reference (LSC + VBS top systems)
    03-foundation-models-2026.md      # 2026 SOTA model picks per subsystem
    04-vietnamese-stack-and-agents.md # Vietnamese stack + agentic retrieval
    05-baseline-2025-analysis.md      # the 2025 baseline repo: what to keep, what to skip
    06-aic2026-dataset-shape.md       # team-channel intel on the 2026 dataset format
  proposals/                          # ARCHITECTURE-level "what we'll build"
    01-interactive-system-architecture.md
    02-automatic-track-agent.md
    03-data-pipeline.md
    04-fine-tuning-plan.md
    05-evaluation-harness.md
    06-ui-ux-design.md
    07-approaches-catalog.md          # all considered approaches, chosen vs skipped
    08-original-contributions.md      # what is actually novel in our system (vs reused SOTA)
    09-llm-path-bakeoff.md            # RTX 5070 local vs Groq cloud: benchmark, criteria, deadline
  specs/                              # COMPONENT-level "exactly how each module behaves"
    template.md                       # SDD spec template
    INDEX.md                          # registry of all specs (status, owner)
    SPEC-0001-evaluation-harness.md   # Approved; Tier 1 shipped
    SPEC-0002-llm-path-bakeoff-runner.md
    SPEC-0018-dres-integration.md     # Draft; borrows from 2025 baseline
  adr/                                # IMMUTABLE record of irreversible decisions
    template.md
    INDEX.md                          # 12 accepted ADRs
    ADR-0001 .. ADR-0012              # see INDEX for the full set
  permissions/                        # explicit-attribution records under ADR-0010
    2025-baseline-reuse.md            # 2025-baseline-author signoff + interview agenda
  datasets/                           # placeholder for dataset references; corpus is gitignored
  papers/                             # 37 downloaded reference papers + LSC SOTA PDF
    foundation-vlm/                   # SigLIP-2, Meta CLIP 2, EVA-CLIP, CLIP, ...
    lsc-systems/                      # LSC review, MERVIN, QUEST-DANTE, EEIoT...
    vbs-systems/                      # VBS 2024/2025 result reports
    vietnamese-multimodal/            # PhoWhisper, PaddleOCR 3.0
    agentic-retrieval/                # SmartRouting, CascadedMM-Agent
    benchmarks/                       # CASTLE, LSC-ADL, Ego4D, EPIC-Kitchens, TimelineQA
  The_State-of-the-Art_in_Lifelog_Retrieval_...2022-2024.pdf  # foundational PDF

.github/
  PULL_REQUEST_TEMPLATE.md            # PR template (asks for SPEC + ADR refs)

eval-results/                         # `bin/eval` output dir; per-run subdirs gitignored
experiments/                          # per-experiment workspaces (e.g. llm-path-bakeoff)
```

## At-a-glance strategy

### Winning hypothesis - floor, edge, moat

**Floor (table stakes - reproduces the 2026 LSC/VBS finalist line):** a Vietnamese-capable multimodal stack (Meta CLIP 2 + SigLIP-2 + InternVideo2 + Vintern-3B-beta + PhoWhisper + PaddleOCR/VietOCR + BGE-M3 in Milvus + Elasticsearch), driven by an LLM planner (SeaLLMs-v3 / Gemini Flash) and a VLM-as-judge reranker. This mirrors patterns from MEMORIA (LSC'25), NII-UIT (VBS'25), and SnapMind (MMM 2026). Necessary but not sufficient - every serious 2026 team will land near here.

**Edge (three original contributions on top of the floor)** - see [`docs/proposals/08-original-contributions.md`](docs/proposals/08-original-contributions.md):
1. **DiacriticBERT** - a Vietnamese late-interaction head trained on a controlled diacritic-noise distribution. Targets the systematic ASR/OCR failure mode that off-the-shelf BGE-M3 ignores.
2. **Per-task-type learned fusion** - replaces the 2009-vintage RRF k=60 default with a per-task LambdaRank model that knows when to favour image-text vs OCR vs ASR. Auto-falls back to RRF if it regresses.
3. **Agent self-distillation** - the interactive-track operator's traces become the training corpus for the automatic-track planner via DSPy. This pattern only exists because 2026 is the first year the automatic track is a serious sub-event.

**Moat (process, not technology):** aggressive operator drills (>=20% of prep time) and a submission-verification panel. PraK1 vs PraK2 differed by 30 points on the same engine; operator skill is the documented lever.

### Architecture in one paragraph
Offline: ingest organiser-provided **keyframes + object detection + YouTube URLs** (see [research-note 06](docs/research-notes/06-aic2026-dataset-shape.md)) -> three image encoders (SigLIP-2, Meta CLIP 2, InternVideo2-1B) into a single multi-vector **Milvus** `keyframes` collection ([SPEC-0006](docs/specs/SPEC-0006-milvus-schema-and-queries.md)), plus a **Qwen3-VL-Embedding-2B offline-only visual-document lane** ([ADR-0012](docs/adr/ADR-0012-qwen-offline-visual-document-lane.md); never the online query encoder); OCR via PaddleOCR/VietOCR and ASR via **yt-dlp YouTube captions** (primary) plus **PhoWhisper** (fallback) into Elasticsearch with BGE-M3 dense+sparse heads + a **DiacriticBERT late-interaction head** (our contribution); YOLO/Places365/LSC-ADL labels as structured filters; organisers' pre-computed CLIP retained as a 4th baseline lane. Online: planner LLM parses the Vietnamese query into a DAG of tool calls, runs in parallel, applies **per-task-type learned fusion** (RRF k=60 fallback), applies structured filters, optionally runs DANTE DP for TRAKE, reranks via Vintern-3B-beta VLM-as-judge, displays top-10 in a React UI; operator confirms via submission-verification bar. The same engine runs the automatic track driven by a planner **distilled from the operator's own traces**.

### Timeline
| Phase | Weeks | Goal |
|---|---|---|
| 0 - Pre-launch | now -> June 15 | Team registered; minimal baseline on LSC'24 |
| 1 - Real-data baseline | June 25 -> mid-July | SigLIP-2 + Meta CLIP 2 on AIC dataset |
| 2 - Full system | mid-July -> mid-Aug | Ensemble + fine-tune + planner + reranker |
| 3 - Preliminary round | Aug | Top-3 to advance |
| 4 - Finals prep | Sept 1-11 | Drills, drills, drills |
| 5 - Finals | Sept 12-26 | Win 1st prize |

## Key technical choices

| Subsystem | Primary | Backup |
|---|---|---|
| Image-text encoder | SigLIP-2 So400m/16@384 + Meta CLIP 2 ViT-H/14 | EVA-02-CLIP-L/14+ |
| Video encoder | InternVideo2-1B | LanguageBind-H |
| Vector DB | Milvus 2.5 hybrid | FAISS HNSW |
| OCR | PaddleOCR PP-OCRv5 + VietOCR fallback | EasyOCR |
| ASR | **yt-dlp YouTube captions (primary)** + PhoWhisper-large + WhisperX (fallback) | Whisper-large-v3-turbo + Gemini diacritic-fix |
| Pre-extracted data | Organisers ship keyframes + OD + baseline CLIP (research-note 06) | -- |
| Text retriever | BGE-M3 (dense+sparse+colbert) | multilingual-e5 |
| VLM reranker | Vintern-3B-beta + Gemini 2.5 Flash | BLIP-2 ITM head |
| Planner LLM | SeaLLMs-v3-7B (local) + Gemini 2.5 Flash | Vistral-7B-FC |
| Agent framework | LangGraph + DSPy | smolagents |
| **Ranker fusion** | **Per-task-type LambdaRank (ours, C2)** | **RRF k=60 (auto-fallback)** |
| **Diacritic robustness** | **DiacriticBERT late-interaction head (ours, C1)** | SeaLLMs-v3 query rewriting |
| **Auto-track planner training** | **DSPy self-distillation on operator traces (ours, C4)** | Zero-shot SeaLLMs-v3 |
| UI | React + Tailwind + shadcn/ui | -- |

## Reading order if you're new to the team

1. [`docs/strategy/00-master-strategy.md`](docs/strategy/00-master-strategy.md) - the master plan
2. [`CONTRIBUTING.md`](CONTRIBUTING.md) + [`AGENTS.md`](AGENTS.md) - SDD workflow (humans + agents)
3. [`docs/proposals/08-original-contributions.md`](docs/proposals/08-original-contributions.md) - what is *actually* novel in our system vs the reused 2026 SOTA floor; read this before you internalise the rest as "the plan"
4. [`docs/research-notes/01-aic-hcmc-prior-editions.md`](docs/research-notes/01-aic-hcmc-prior-editions.md)
5. [`docs/research-notes/05-baseline-2025-analysis.md`](docs/research-notes/05-baseline-2025-analysis.md) - the 2025 baseline repo: what we borrow, what we replace
6. [`docs/research-notes/06-aic2026-dataset-shape.md`](docs/research-notes/06-aic2026-dataset-shape.md) - what the 2026 dataset looks like (provisional)
7. [`docs/research-notes/02-lsc-vbs-systems-deep-dive.md`](docs/research-notes/02-lsc-vbs-systems-deep-dive.md)
8. [`docs/proposals/01-interactive-system-architecture.md`](docs/proposals/01-interactive-system-architecture.md)
9. [`docs/proposals/02-automatic-track-agent.md`](docs/proposals/02-automatic-track-agent.md)
10. [`docs/specs/INDEX.md`](docs/specs/INDEX.md) + [`docs/adr/INDEX.md`](docs/adr/INDEX.md) - what's been specified and what's been decided
11. The rest in `docs/proposals/` and `docs/research-notes/` as you implement.
12. Skim abstracts in `docs/papers/` before deep-reading any one.

## Getting started (dev)

The evaluation harness is shipped; you can run it against a deterministic stub backend today.

```
uv sync --dev                                       # install deps + dev tools
uv run pytest -q                                    # 36 tests, ~1s
uv run eval --tasks tests/mock_tasks/smoke_20.jsonl --system my-experiment --no-latency-sim
```

The smoke run produces `eval-results/my-experiment/<utc-timestamp>/{report.html, metrics.json, README.md}`. See [SPEC-0001](docs/specs/SPEC-0001-evaluation-harness.md) for the full contract; the 20-task smoke corpus is hand-written placeholder content, not the real (post-June-25) competition data.

Lint + format gates (run before opening a PR):

```
uv run ruff format --check .
uv run ruff check .
```

## How to contribute

This repo follows **Spec-Driven Development**. The full workflow is in [`CONTRIBUTING.md`](CONTRIBUTING.md); AI assistants additionally read [`AGENTS.md`](AGENTS.md). The one-line rule: **no code without a spec, no decision without an ADR**.

Quick path for a new feature:
1. Find or write the spec in [`docs/specs/`](docs/specs/) using [`docs/specs/template.md`](docs/specs/template.md). Reserve an ID in [`docs/specs/INDEX.md`](docs/specs/INDEX.md).
2. If the spec depends on a non-obvious irreversible decision, write the ADR first using [`docs/adr/template.md`](docs/adr/template.md).
3. Branch as `spec/NNNN-short-name`. Code in [`src/aic2026/`](src/aic2026/) references the spec ID at the top of each file. Tests in [`tests/`](tests/) are named for the acceptance criteria they prove (e.g. `test_..._AC2`).
4. Open a PR using [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md). Attach `bin/eval` evidence if score-relevant.

Every PR runs the smoke eval set (20 mock tasks); nightly runs the full eval set (300 mock tasks once authored); regressions >2% block merge.

## License

Internal team artefact. Cited papers are property of their respective authors; consult each PDF for its license.
