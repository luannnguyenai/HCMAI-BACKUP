# AIC2026 - Master Strategy: Targeting 1st Prize (Bang A)

> Author intent: A single end-to-end document that any team member can read and understand (a) what we're building, (b) why, (c) how, and (d) when. Backing detail lives in `docs/research-notes/` and `docs/proposals/`.

**Competition:** Hoi thi Thu thach Tri tue Nhan tao (AI-Challenge) TP.HCM 2026 - <https://aichallenge.hochiminhcity.gov.vn/>
**Problem:** Intelligent virtual assistant for analysing and retrieving information from a large multimedia (image + audio + text) dataset.
**Format:** LSC/VBS-style - Known-Item Search + Question Answering + TRAKE (4-scene temporal alignment) + Ad-hoc, with two evaluation tracks (interactive + **NEW** automatic agent).

---

## 1. Critical dates (checked 2026-06-24)

| Date | Milestone | Implication for us |
|---|---|---|
| **May 15-20, 2026** | Competition launch | Rules and public framing confirmed on the official site |
| **June 15, 2026** | Registration closes | Past deadline; team status must be confirmed outside this repo |
| **June 25, 2026** | Preliminary round content + dataset released | Immediate Phase 1 trigger; read rules and sample data within 24 hours |
| **June-July 2026** | Training sessions by organisers | Attend all sessions; capture rule/data clarifications in research notes |
| **Aug 2026** | Preliminary round runs | First competitive test |
| **Aug 30, 2026** | Preliminary results announced | Either advance or stop |
| **Sept 12-26, 2026** | Finals - interactive on-site (**T-16 weeks**) | Real prize |

**Status date: 2026-06-24.** The repo has crossed from Phase 0 planning into
a one-day Phase 1 readiness checkpoint. The official schedule still makes the
strategy feasible, but the remaining runway is compressed: the June 25
dataset/rules release must become a focused ingestion, indexing, UI, and DRES
sprint. See [`01-feasibility-audit-2026-06-24.md`](01-feasibility-audit-2026-06-24.md).

Source: official competition page, section "Thoi gian, tien do du kien trien
khai cac vong thi" and "Noi dung", <https://aichallenge.hochiminhcity.gov.vn/>,
checked 2026-06-24.

### 1.1 Feasibility snapshot as of 2026-06-24

**Verdict: feasible, but compressed.** The strongest evidence in favour is the
repo's SDD discipline, working eval harness, remote GPU/R2 substrate, Milvus
indexing path, MVP serving/UI loop, and mature C1 DiacriticBERT workstream. The
strongest risk is that the actual competition-facing retrieval product is not
yet end-to-end: data ingestion, Elasticsearch/text retrieval, DRES submit,
submission verification, planner, reranker, and operator traces still need to
close the loop.

Immediate priorities:

1. Turn the June 25 dataset release into SPEC-0003 plus a sample loader within
   48 hours.
2. Build a thin real-data retrieval baseline on the existing Milvus path before
   expanding model breadth.
3. Ship DRES submission, submission verification, and trace logging early
   enough for operator practice and C4 self-distillation.
4. Keep C3 and C5 out of scope until Phase 1 gates are green.

## 2. Winning hypothesis - floor, edge, moat

We do not believe a parts-list of 2026 SOTA models wins this competition. Every serious team will read the same LSC SOTA review, the same MMM 2026 SnapMind paper, the same MEMORIA writeup. They will land near the same stack. We need a *floor* (don't lose to integration mistakes), an *edge* (original technical contributions), and a *moat* (process advantages compounding edge into score).

### 2.1 Floor - reproducing the 2026 SOTA Vietnamese stack

The floor is necessary but not sufficient. It is everything in proposals 01-07:

- **Vietnamese-capable multimodal stack.** Meta CLIP 2 ViT-H/14 (best XM3600 multilingual) + Vintern-3B-beta (Vi-first VLM) + PhoWhisper-large + PaddleOCR PP-OCRv5 + VietOCR + BGE-M3. *("Vietnamese-capable" not "Vietnamese-native" - Meta CLIP 2 is multilingual, not Vi-first; only Vintern is Vi-first.)*
- **3-model image-text + video ensemble** (SigLIP-2 + Meta CLIP 2 + InternVideo2-1B) in **Milvus hybrid** with structured filters. This pattern is the LSC/VBS top-3 default since 2022. MEMORIA (LSC'25) reported their win-attribution as the FAISS->Milvus swap; we cite this as motivating evidence, not as a controlled ablation.
- **LLM-driven planner for both tracks** (SeaLLMs-v3-7B local + Gemini 2.5 Flash escalation) matching the SnapMind (MMM 2026) blueprint. The unification is engineering reuse, not a research contribution.
- **VLM-as-judge reranker** (Vintern-3B-beta + Gemini 2.5 Flash).
- **DANTE-style DP for TRAKE** (AIO_Owlgorithms LSC'25).
- **Speed-optimised UX**: keyframe scrubbing (diveXplore), TRAKE drag-drop (VISIONE-flavoured), temporal `q1 < q2` syntax, query history, and the **submission-verification panel** (the documented PraK lesson).

This is the table-stakes line for a 2026 finalist. We will not pretend it differentiates us.

### 2.2 Edge - three original contributions (see [`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md))

On top of the floor, three workstreams that no LSC/VBS team has shipped in the public literature:

1. **DiacriticBERT (C1)** - a small late-interaction head trained on a controlled Vietnamese diacritic-noise schedule. Targets the systematic ASR/OCR failure mode (item 3 in our own SS 7 list of things-that-lose-the-competition). Expected lift: +2-5% R@1 on OCR/ASR-bridged queries. Risk: bounded - it is a head, not a backbone. Owner: Vietnamese NLP Engineer. Effort: ~1 week.

2. **Per-task-type learned fusion (C2)** - replaces uniform RRF k=60 with a LambdaRank model selected at query time by the planner's task_type emit. RRF ignores ranker quality across the 12-15 heterogeneous lists we fuse. Expected lift: +1-3% NDCG@10. Risk: trivial - runtime auto-fallback to RRF if the learned model is worse on a held-out slice. Owner: Lead Engineer. Effort: ~3 days.

3. **Agent self-distillation (C4)** - the interactive-track operator's correct, fast submissions become the training corpus for the automatic-track planner via DSPy MIPRO. This pattern only exists because AIC2026 is the first year the automatic track is a serious sub-event - SnapMind gives the architecture, not the training signal. Expected lift: +10-20% automatic-track R@1 vs zero-shot planner. Owner: Operator-1 / ML Engineer. Effort: ~1 week after instrumentation.

Two further backup contributions (C3 PriorDP for TRAKE, C5 Counterfactual VLM rerank) are scoped in proposal 08 but ship only if Phase 2 has slack.

### 2.3 Moat - operator drills and the verification panel

LSC review SS IV-D quantifies this: PraK1 and PraK2 differed by 30 points on the same engine, purely from operator skill. We will spend at least **20% of total preparation time** on team-internal practice runs with mock queries, and we will ship the submission-verification panel as a first-class anti-foot-gun. This is a *process* moat, not a technical one - every team can in principle do it; few will.

### 2.4 The honest summary

If at least two of {C1, C2, C4} pass their dev-set ablations, we are a technically differentiated finalist with a clear story for the press kit and the post-competition paper. If only one passes, we are partially differentiated. If none pass, we are a competent but undifferentiated finalist - which is still defensible given the floor, but we should expect a tighter race.

## 3. Architecture at a glance

```
+----------------------------------------------------------------+
| OFFLINE PIPELINE (one-time + nightly refresh)                  |
|                                                                |
| raw videos / images / audio                                    |
|        |                                                       |
|        +-- TransNetV2 shot detection                           |
|        |     + KDE-GMM frame sampling -> ~1M keyframes         |
|        |                                                       |
|        +-- SigLIP-2 So400m/16@384 ---- 1152-d ---> Milvus#1    |
|        +-- Meta CLIP 2 ViT-H/14   ---- 1024-d ---> Milvus#2    |
|        +-- InternVideo2-1B (4 frames/clip) ------> Milvus#3    |
|        |                                                       |
|        +-- PaddleOCR PP-OCRv5 (Vi) + VietOCR fallback          |
|        |     -> text + boxes ---> Elasticsearch + BGE-M3       |
|        |                                                       |
|        +-- PhoWhisper-large + WhisperX timestamps              |
|        |     -> text + word-level ts ---> Elastic + BGE-M3     |
|        |                                                       |
|        +-- LAION-CLAP audio events ---------> Milvus#4 (sound) |
|        |                                                       |
|        +-- Qwen2.5-VL-7B captioning -> long Vietnamese caps    |
|        |     ---> Elasticsearch + BGE-M3                       |
|        |                                                       |
|        +-- YOLOv8 objects + Places365 scenes + LSC-ADL labels  |
|        |     ---> Milvus structured fields                     |
|        |                                                       |
|        +-- Time/location metadata ---> Milvus structured       |
|                                                                |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| ONLINE PIPELINE (per query, target p50<900ms, p95<2s)          |
|                                                                |
| User query (text / image / audio)                              |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | Planner LLM (SeaLLMs-v3-7B local, Gemini 2.5 Flash hard) |  |
| |  - JSON intent: {object, action, where, when, how_many}   |  |
| |  - emits task_type for fusion-model selection (C2)        |  |
| |  - emit DAG of tool calls + fusion weights                |  |
| |  - paraphrase Vi query into 3 variants for ensemble       |  |
| | * Automatic-track: distilled prompt from operator traces  |  |
| |   (C4 - DSPy MIPRO over data/operator_traces.parquet)     |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v   (parallel execution)                                |
| +-----------------------------------------------------------+  |
| | text->image x3 ensemble | OCR BM25+dense+DiacBERT (C1)    |  |
| | image->image SigLIP-2   | ASR BM25+dense+DiacBERT (C1)    |  |
| | sound CLAP              | metadata filter                 |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | Per-task-type learned fusion (C2; LightGBM LambdaRank)    |  |
| |  - RRF k=60 auto-fallback if learned model regresses      |  |
| |  -> top-200                                               |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | Structured filters: time/location/object/ADL -> top-50    |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v   (only for TRAKE queries)                            |
| +-----------------------------------------------------------+  |
| | DANTE DP (default) or PriorDP (C3 if shipped)             |  |
| | 4-scene temporal alignment, lambda in [.001,.01]          |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | VLM-as-judge rerank (Vintern-3B-beta / Gemini 2.5 Flash)  |  |
| |  3x3 grid + CoT; position-bias mitigation: 3-vote shuffle |  |
| |  C5 counterfactual pruning mode for OOK queries (backup)  |  |
| |  -> top-10                                                |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| Interactive: results -> React UI -> operator -> DRES submit    |
|              + trace logger -> data/operator_traces.parquet    |
| Automatic:   confidence agent decides submit / retry / fallback|
|                                                                |
+----------------------------------------------------------------+

Items marked C1/C2/C4 are our original contributions (proposal 08).
The C4 trace logger closes the loop: the interactive operator's
correct, fast submissions become the automatic agent's training corpus.
```

For the full diagram and component decisions, see `docs/proposals/01-interactive-system-architecture.md` and `docs/proposals/02-automatic-track-agent.md`.

> Note (updated 2026-06-05): the ASCII sketch above predates two shipped
> decisions. (1) Storage is a **single multi-vector Milvus `keyframes`
> collection** with named dense fields (SigLIP-2 1152, Meta CLIP 2 1024,
> Qwen3-VL-Embedding-2B 2048), keyed by a global
> `pk = "<video_id>_<frame_id>"`, per
> [SPEC-0006](../specs/SPEC-0006-milvus-schema-and-queries.md). This
> supersedes the per-encoder `Milvus#1..#4` boxes drawn here. (2) An additional
> **Qwen3-VL-Embedding-2B offline-only visual-document lane** is indexed
> alongside the floor encoders ([ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md);
> offline `encode_image` only, never the online query encoder). Encoder dims
> are verified: SigLIP-2 = 1152, Meta CLIP 2 = 1024,
> Qwen3-VL-Embedding-2B = 2048, InternVideo2 = 768.

## 4. Updated 17-week plan from the June 24 audit

### Phase 0 - Pre-launch alignment (closed: May 15 -> June 15)
- [ ] Confirm team registration status externally; the repo cannot prove this.
- [x] Strategy, proposals, specs, and ADR workflow are in place.
- [x] Evaluation harness, remote runner, R2 artifact path, and C1 DiacriticBERT
  tooling are implemented enough to support Phase 1.
- [ ] Record any organiser license clarification in `docs/research-notes/` or a
  follow-up ADR if it changes the model stack.

### Phase 1 - Real-data baseline (June 25 -> mid-July) - 3 weeks
- [ ] Within 24 hours of release: summarise official rules, task types, dataset
  shape, scoring, and submission API deltas.
- [ ] Approve SPEC-0003 and run a sample loader over organiser keyframes,
  metadata, object detection, YouTube URLs, and any provided CLIP embeddings.
- [ ] Build a thin retrieval baseline first: organiser metadata + one SigLIP-2
  image lane + local top-10 report.
- [ ] Add OCR/ASR/description text lanes and a minimal Milvus/Elasticsearch or
  local fallback index.
- [ ] Wire DRES or organiser scoring submission via SPEC-0018.
- [ ] Build the React console MVP with grid, neighbour inspection, submission
  verification, and trace logging.
- [ ] **Decision gate**: by mid-July, a real-data baseline must support search,
  inspect, submit, trace, and nightly smoke evaluation.

### Phase 2 - Full system + fine-tuning + original contributions (mid-July -> mid-Aug) - 4 weeks

**Floor work** (reused 2026 SOTA):
- [ ] DreamLIP-style synthetic-caption generation with Qwen2.5-VL-72B INT4 -> LoRA fine-tune SigLIP-2 on synthetic Vi pairs.
- [ ] Train Vietnamese ColVintern late-interaction head on competition data.
- [ ] Add VLM-as-judge reranker (Vintern-3B-beta), with position-bias mitigation (3-vote ensemble + input shuffle).
- [ ] Add DANTE DP for TRAKE.
- [ ] Add LLM planner (SeaLLMs-v3-7B) for both tracks.
- [ ] Add submission-verification panel (anti-foot-gun for the operator).

**Edge work** (original contributions - see [`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md)):
- [ ] **C1 - DiacriticBERT**: build the controlled diacritic-noise function, build the contrastive corpus (~2M pairs), train the late-interaction head, ablate. Ship if pass.
- [ ] **C2 - Per-task-type learned fusion**: build LightGBM LambdaRank harness, train per-task models, wire planner-driven model selection, A/B against RRF k=60 with auto-fallback guardrail. Ship if pass.
- [ ] **C4 - Agent self-distillation**: ensure operator-trace logging is live since Phase 1; run DSPy MIPRO round 1 on Phase 1 traces; ablate distilled vs zero-shot planner. Ship if pass.
- [ ] **C3 - PriorDP** (only if Phase 2 slack and TRAKE still in 2026): build scene-transition prior matrix; replace DANTE linear penalty with prior-weighted DP; ablate.
- [ ] **C5 - Counterfactual VLM rerank** (only if Phase 2 slack): build the iterative-pruning rerank loop; ablate on the long-tail slice.

**Decision gate**: full system beats baseline by 20%+ on internal practice AND
at least 2 of {C1, C2, C4} pass their ablations (or are documented as negative
results with their fallback shipped). If Phase 1 gates slip, defer C3/C5 and
ship reliability over novelty breadth.

### Phase 3 - Preliminary round (Aug) - 4 weeks
- [ ] Compete in preliminary. Goal: top-3 finish guarantees a finals slot.
- [ ] During and after: analyse every mistake; instrument the planner & reranker.
- [ ] **Decision gate**: advance to finals (Aug 30 announcement).

### Phase 4 - Finals preparation (Sept 1-11) - ~2 weeks
- [ ] Hardware shake-down on the venue setup (latency, network, projector contrast).
- [ ] Operator drills - at least 30 mock queries with all 3 team operators.
- [ ] Prepare contingency: offline-mode if internet flaky at the venue.
- [ ] **Decision gate**: 90%+ accuracy on internal mock-finals.

### Phase 5 - Finals (Sept 12-26)
- [ ] Compete; iterate between rounds.

## 5. Team structure (recommended)

For a 5-person team (max per rules):

1. **Lead Engineer / Retrieval Architect** - owns the embedding/index/fusion pipeline.
2. **Vietnamese NLP Engineer** - owns OCR, ASR, transcript cleanup, Vietnamese embeddings, query rewriting prompts.
3. **Frontend / UX Engineer** - owns the React UI, keyframe scrubber, TRAKE drag-drop, submission verification.
4. **Operator-1 / ML Engineer** - primary operator on competition day; owns the planner LLM prompts, VLM reranker, fine-tuning.
5. **Operator-2 / DevOps + Data** - secondary operator; owns the data pipeline (TransNetV2, captioning, indexing), DRES integration, monitoring.

The two operators must be cross-trained and able to swap mid-round.

## 6. Hardware plan

**Development**: local laptops/workstations for CPU-safe development; remote
GH200/H200-style leases for embedding extraction, captioning, C1/C2/C4
training, and quantisation calibration. Cloud burst is managed through
`bin/remote` and Cloudflare R2 per ADR-0011.

**Finals**: bring 2 identical laptops (mirror images), redundant power, 4G hotspot backup. Pre-stage all model weights on each laptop (Vintern, PhoWhisper, SigLIP-2 quantised, Meta CLIP 2 quantised - total ~30 GB).

## 7. The eight things that will lose this competition if we skip them

1. **Submission-verification panel.** Every mistake costs 10 points. PraK1 vs PraK2 differed by 30 points on the same engine.
2. **TRAKE DP.** Pure semantic search scatters the correct 4-scene sequence across non-adjacent frames. Without DANTE-style DP we lose TRAKE points.
3. **Vietnamese diacritic-noise handling at retrieval.** ASR (Whisper-large-v3, even PhoWhisper) and OCR systematically corrupt diacritics. Either post-process at index time with PhoWhisper + Gemini diacritic correction, *or* train the retriever to absorb the noise distribution itself. **Both are in scope**: post-processing in proposal 01 SS 5.6, and DiacriticBERT (C1) as our primary original contribution in proposal 08.
4. **OOK entity handling.** Frozen CLIP can't embed local Vietnamese celebrities/brands. LLM query rewrite + external image search (proposal 02 SS 10.2), plus C5 counterfactual rerank as a backup contribution, are the mitigations.
5. **Operator practice.** Allocate >=20% of total prep time to mock runs. This is also the training corpus for C4 (agent self-distillation).
6. **Multi-model ensemble.** Single-model systems plateaued in 2023. The 2025 LSC winner attributed its win to swapping a single component (graph-DB -> Milvus); treat that as motivation, not as a proven causal claim. Build the ensemble from day 1.
7. **Verification of license compatibility.** PhoWhisper is CC-BY-NC. Confirm with organisers in writing before depending on it.
8. **Internal evaluation harness.** Don't measure progress by feelings. Write a fixed test set of mock queries and run nightly regression. The harness is *also* the gate for the C1/C2/C4 ablations; without it, the original contributions can't be shipped responsibly.

## 8. The five things we will deliberately *not* do

1. **VR/AR/eye-tracking.** Cool but adds operator cognitive load and hardware risk. Skip CollaXRSearch / EAGLE / vitrivr-VR style features.
2. **Knowledge graphs.** LifeGraph is academically interesting but slow and brittle. Use Milvus + structured fields instead.
3. **Custom model training from scratch.** All fine-tuning is LoRA + synthetic caption data; never train a backbone from scratch. (C1 trains a small 2-layer projection head on top of a *frozen* BGE-M3; this is not "from scratch" in the spirit of this rule.)
4. **More than 3 embedding models in the ensemble.** Diminishing returns; harder to debug. (DiacriticBERT is a head over BGE-M3, not a separate backbone; it does not violate this rule.)
5. **Pure-LLM end-to-end ("just throw it at GPT-4o").** LSC review SS IV-D: top QA systems were *not* purely conversational LLM. RAG works; chat-only does not.

## 9. Reading list (in priority order)

1. `docs/strategy/00-master-strategy.md` (this document)
2. **`docs/proposals/08-original-contributions.md`** - what is novel in our system vs the reused 2026 SOTA floor; read this before internalising the rest as "the plan"
3. `docs/research-notes/01-aic-hcmc-prior-editions.md` - what worked at this competition
4. `docs/research-notes/02-lsc-vbs-systems-deep-dive.md` - the world reference
5. `docs/proposals/01-interactive-system-architecture.md` - what we're building
6. `docs/proposals/02-automatic-track-agent.md` - the agent track
7. `docs/research-notes/03-foundation-models-2026.md` - model choices
8. `docs/research-notes/04-vietnamese-stack-and-agents.md` - Vietnamese specifics
9. `docs/proposals/03-data-pipeline.md`, `04-fine-tuning-plan.md`, `05-evaluation-harness.md`, `06-ui-ux-design.md`, `07-approaches-catalog.md`
10. The LSC SOTA review PDF in `docs/papers/lsc-systems/` (already there)
11. The downloaded papers in `docs/papers/` (skim abstracts; deep-read only when implementing)

## 10. Open questions / decisions to make next

1. **Team registration status**: June 15 has passed; confirm externally that the
   team is registered and record only non-sensitive status in repo docs.
2. **License confirmation**: email/Slack the organising committee to confirm whether CC-BY-NC weights (PhoWhisper, jina-v3) are allowed. **Risk reduced (2026-05-28)**: team-channel intel ([research-note 06](../research-notes/06-aic2026-dataset-shape.md) §2.1) confirms organiser metadata carries YouTube URLs, so `yt-dlp` auto-captions become the primary Vietnamese-transcript source and PhoWhisper is demoted to a fallback for videos lacking captions. Empirical coverage must be measured after the June 25 data release.
3. **Hardware** (resolved May 25, see [ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md) and [ADR-0006](../adr/ADR-0006-int4-quantization-hot-path.md)): finals box = **RTX 5070 (12 GB)**; offline training/indexing on **GH200** cloud burst. Implication: all hot-path LLM/VLM weights must be quantized to INT4 or FP4; only text-tower encoders kept online; image embeddings pre-computed on GH200.
4. **LLM path** (in flight, see [ADR-0005](../adr/ADR-0005-llm-path-bakeoff-gates-planner.md) + [proposal 09](../proposals/09-llm-path-bakeoff.md) + [SPEC-0002](../specs/SPEC-0002-llm-path-bakeoff-runner.md)): local-5070 vs Groq-cloud bakeoff. Owner = team lead. Deadline = end of June. Closes the original "cloud budget" question with a measurement, not a guess.
5. **UI stack** (resolved May 25, see [ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md)): no Streamlit / Gradio. React 18 + Vite + Zustand + WebSocket + virtualised grid + nginx-served thumbnails. Latency SLO: p95 end-to-end < 2 s, p50 < 900 ms.
6. **Workflow** (resolved May 26, see [ADR-0009](../adr/ADR-0009-sdd-workflow.md) + [`CONTRIBUTING.md`](../../CONTRIBUTING.md)): **Spec-Driven Development**. Every code change traces to a SPEC; every irreversible decision is recorded as an ADR.
7. **Dataset release validation**: on June 25, validate the six follow-up
   data-shape questions in [research-note 06](../research-notes/06-aic2026-dataset-shape.md)
   and update SPEC-0003 before writing importer code.
8. **TRAKE in 2026?**: not confirmed yet that TRAKE remains a task in 2026; design for it but keep the rest task-agnostic.
9. **Operator candidates and drills**: identify the 2 operators immediately and
   start practice as soon as the React MVP can search and submit. Natural
   candidate: [`ThanhToan2111`](https://github.com/ThanhToan2111) (current team
   member, author of the 2025 baseline) has the most operator-side experience
   with DRES and the AIC HCMC venue.
10. **2025 baseline interview**: the 30-minute agenda in
    [`docs/permissions/2025-baseline-reuse.md`](../permissions/2025-baseline-reuse.md)
    SS 4 remains high leverage for SPEC-0018 and UI/operator lessons; schedule
    it before DRES integration work.
