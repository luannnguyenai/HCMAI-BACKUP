# AIC2026 - SCOAI

> Strategy, research, and architecture for competing in the **AI Challenge HCMC 2026** (Hoi thi Thu thach Tri tue Nhan tao TP.HCM 2026), targeting **1st prize in Bang A** (university division).

**Competition site**: <https://aichallenge.hochiminhcity.gov.vn/>
**Today**: May 24, 2026 (T-17 weeks to finals)
**Problem**: Intelligent virtual assistant for retrieving information from large multimedia data (image + audio + text); format inherits from international **Lifelog Search Challenge (LSC)** and **Video Browser Showdown (VBS)** with a NEW **automatic-agent track** in 2026.

---

## Read this first

[`docs/strategy/00-master-strategy.md`](docs/strategy/00-master-strategy.md) - one-page summary of our plan, timeline, team, and the eight things that will lose the competition if we skip them.

---

## What's in this repo (so far)

```
docs/
  strategy/
    00-master-strategy.md            # the master plan
  research-notes/                    # external research, every claim cited
    01-aic-hcmc-prior-editions.md    # 2023/24/25 lessons
    02-lsc-vbs-systems-deep-dive.md  # world reference (LSC + VBS top systems)
    03-foundation-models-2026.md     # 2026 SOTA model picks per subsystem
    04-vietnamese-stack-and-agents.md # Vietnamese stack + agentic retrieval
  proposals/                         # concrete architecture proposals
    01-interactive-system-architecture.md
    02-automatic-track-agent.md
    03-data-pipeline.md
    04-fine-tuning-plan.md
    05-evaluation-harness.md
    06-ui-ux-design.md
    07-approaches-catalog.md         # all considered approaches, chosen vs skipped
  papers/                            # 37 downloaded reference papers
    foundation-vlm/                  # SigLIP-2, Meta CLIP 2, EVA-CLIP, CLIP, ...
    lsc-systems/                     # LSC review, MERVIN, QUEST-DANTE, EEIoT...
    vbs-systems/                     # VBS 2024/2025 result reports
    vietnamese-multimodal/           # PhoWhisper, PaddleOCR 3.0
    agentic-retrieval/               # SmartRouting, CascadedMM-Agent
    benchmarks/                      # CASTLE, LSC-ADL, Ego4D, EPIC-Kitchens, TimelineQA
  The_State-of-the-Art_in_Lifelog_Retrieval_...2022-2024.pdf  # foundational PDF
```

## At-a-glance strategy

### Winning hypothesis
Four advantages stacked together:
1. **Vietnamese-native multimodal stack** (Meta CLIP 2 + Vintern + PhoWhisper + PaddleOCR/VietOCR + BGE-M3)
2. **3-model image-text ensemble** with RRF fusion in Milvus hybrid index
3. **LLM-driven planner** (SeaLLMs-v3 / Gemini Flash) that drives both interactive and automatic tracks
4. **Speed-optimised UX** with keyframe scrubbing, temporal `q1 < q2`, TRAKE drag-drop palette, and a submission-verification panel

### Architecture in one paragraph
Offline: TransNetV2 shot detection -> keyframes -> three image encoders (SigLIP-2, Meta CLIP 2, InternVideo2-1B) into Milvus; PaddleOCR/VietOCR + PhoWhisper into Elasticsearch with BGE-M3 dense+sparse heads; YOLO/Places365/LSC-ADL labels as structured filters. Online: planner LLM parses Vietnamese query into a DAG of tool calls, runs in parallel, RRF-fuses, applies structured filters, optionally runs DANTE DP for TRAKE, reranks via Vintern-3B-beta VLM-as-judge, displays top-10 in a React UI; operator confirms via submission-verification bar.

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
| ASR | PhoWhisper-large + WhisperX | Whisper-large-v3-turbo + Gemini diacritic-fix |
| Text retriever | BGE-M3 (dense+sparse+colbert) | multilingual-e5 |
| VLM reranker | Vintern-3B-beta + Gemini 2.5 Flash | BLIP-2 ITM head |
| Planner LLM | SeaLLMs-v3-7B (local) + Gemini 2.5 Flash | Vistral-7B-FC |
| Agent framework | LangGraph + DSPy | smolagents |
| UI | React + Tailwind + shadcn/ui | -- |

## Reading order if you're new to the team

1. [`docs/strategy/00-master-strategy.md`](docs/strategy/00-master-strategy.md)
2. [`docs/research-notes/01-aic-hcmc-prior-editions.md`](docs/research-notes/01-aic-hcmc-prior-editions.md)
3. [`docs/research-notes/02-lsc-vbs-systems-deep-dive.md`](docs/research-notes/02-lsc-vbs-systems-deep-dive.md)
4. [`docs/proposals/01-interactive-system-architecture.md`](docs/proposals/01-interactive-system-architecture.md)
5. [`docs/proposals/02-automatic-track-agent.md`](docs/proposals/02-automatic-track-agent.md)
6. The rest in `docs/proposals/` and `docs/research-notes/` as you implement.
7. Skim abstracts in `docs/papers/` before deep-reading any one.

## How to contribute

This repo currently holds strategy and research. Code lives in (forthcoming) `src/`, `infra/`, `train/`. When implementation starts in Phase 1 (after June 25):
- Each proposal becomes a tracking issue.
- Each module has an owner (see team table in proposal 01).
- Every PR runs the smoke eval set (20 mock tasks).
- Nightly runs the full eval set (300 mock tasks); regressions >2% block merge.

## License

Internal team artefact. Cited papers are property of their respective authors; consult each PDF for its license.
