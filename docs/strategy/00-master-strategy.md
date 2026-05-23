# AIC2026 - Master Strategy: Targeting 1st Prize (Bang A)

> Author intent: A single end-to-end document that any team member can read and understand (a) what we're building, (b) why, (c) how, and (d) when. Backing detail lives in `docs/research-notes/` and `docs/proposals/`.

**Competition:** Hoi thi Thu thach Tri tue Nhan tao (AI-Challenge) TP.HCM 2026 - <https://aichallenge.hochiminhcity.gov.vn/>
**Problem:** Intelligent virtual assistant for analysing and retrieving information from a large multimedia (image + audio + text) dataset.
**Format:** LSC/VBS-style - Known-Item Search + Question Answering + TRAKE (4-scene temporal alignment) + Ad-hoc, with two evaluation tracks (interactive + **NEW** automatic agent).

---

## 1. Critical dates (from today, May 24, 2026)

| Date | Milestone | Implication for us |
|---|---|---|
| **May 15-20, 2026** | Competition launch (just happened) | Confirm we have the latest official rules |
| **June 15, 2026** | Registration closes (**T-3 weeks**) | Register team before this |
| **June 25, 2026** | Preliminary round content + dataset released (**T-4 weeks**) | Start building on real data |
| **June-July 2026** | Training sessions by organisers (**~8 weeks**) | Attend all of them; great signal source |
| **Aug 2026** | Preliminary round runs | First competitive test |
| **Aug 30, 2026** | Preliminary results announced | Either advance or stop |
| **Sept 12-26, 2026** | Finals - interactive on-site (**T-16 weeks**) | Real prize |

**Today T-0 is May 24, 2026.** We have **roughly 17 weeks**. Tight but tractable if we start in parallel.

## 2. Winning hypothesis

We will win 1st prize in Bang A by combining four advantages:

1. **A Vietnamese-native multimodal stack.** Most LSC/VBS systems use English-optimised CLIP variants. We use **Meta CLIP 2 ViT-H/14** (best multilingual XM3600) + **Vintern-3B-beta** (Vietnamese-native VLM) + **PhoWhisper-large** + **PaddleOCR Vietnamese + VietOCR** + **BGE-M3 / bkai-vietnamese-bi-encoder** for transcript/OCR text. This is provably what the 2025 best Vietnamese teams (MERVIN, NII-UIT VBS'25 winner) built, plus we upgrade to 2026 frontier models. *(See `docs/research-notes/04-vietnamese-stack-and-agents.md`.)*

2. **A 3-model ensemble with smart fusion.** SigLIP-2 (English-strong default), Meta CLIP 2 (Vietnamese), and a video-temporal encoder (InternVideo2-1B). Late fusion via **Reciprocal Rank Fusion (k=60)**. Indexed in **Milvus hybrid (dense + structured filter)**. This mirrors what every LSC/VBS top-3 system has done since 2022, and what specifically MEMORIA (LSC'25 winner) attributed its win to. *(See `docs/proposals/01-interactive-system-architecture.md`.)*

3. **An LLM-driven planner for both interactive and automatic tracks.** A single planner LLM (SeaLLMs-v3-7B function-calling for the bulk; Gemini 2.5 Flash for hard queries) decomposes Vietnamese natural-language queries into tool calls. The same planner serves both tracks: it advises the human operator in the interactive track, and runs unattended in the automatic track. This unifies code, makes the automatic track essentially free given the interactive system, and matches the **SnapMind (MMM 2026)** blueprint. *(See `docs/proposals/02-automatic-track-agent.md`.)*

4. **A laser-focused UX optimised for speed.** Activity-clustered timeline, group-by-day grid, **keyframe scrubbing** (diveXplore-style), temporal `q1 < q2` syntax, +/- relevance feedback, **TRAKE submission palette** with drag-and-drop 4-scene ordering, query history, and a **submission-verification panel** that prevents wasted submissions. *(See `docs/proposals/06-ui-ux-design.md`.)*

**The single bet that differentiates us from middle-of-pack teams:** investment in **operator training and the verification panel**. LSC review SS IV-D quantifies this - PraK1 vs PraK2 differed by 30 points on the same engine purely from operator skill. We will spend at least 20% of our preparation time on team-internal practice runs with mock queries.

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
|        +-- SigLIP-2 So400m/16@384 ---- 1024-d ---> Milvus#1    |
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
| ONLINE PIPELINE (per query, target p50<800ms, p95<2s)          |
|                                                                |
| User query (text / image / audio)                              |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | Planner LLM (SeaLLMs-v3-7B local, Gemini 2.5 Flash hard) |  |
| |  - JSON intent: {object, action, where, when, how_many}   |  |
| |  - emit DAG of tool calls + fusion weights                |  |
| |  - paraphrase Vi query into 3 variants for ensemble       |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v   (parallel execution)                                |
| +-----------------------------------------------------------+  |
| | text->image x3 ensemble | OCR BM25+dense | ASR BM25+dense |  |
| | image->image SigLIP-2   | sound CLAP     | metadata filter|  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | Score normalisation + RRF (k=60) -> top-200               |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | Structured filters: time/location/object/ADL -> top-50    |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v   (only for TRAKE queries)                            |
| +-----------------------------------------------------------+  |
| | DANTE DP: 4-scene temporal alignment, lambda in [.001,.01]|  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| +-----------------------------------------------------------+  |
| | VLM-as-judge rerank (Vintern-3B-beta / Gemini 2.5 Flash)  |  |
| |  3x3 grid prompt, CoT, output top-10                      |  |
| +-----------------------------------------------------------+  |
|        |                                                       |
|        v                                                       |
| Interactive: results -> React UI -> operator -> DRES submit    |
| Automatic:   confidence agent decides submit / retry / fallback|
|                                                                |
+----------------------------------------------------------------+
```

For the full diagram and component decisions, see `docs/proposals/01-interactive-system-architecture.md` and `docs/proposals/02-automatic-track-agent.md`.

## 4. The 17-week plan

### Phase 0 - Pre-launch alignment (Now -> June 15) - 3 weeks
- [ ] Team registered before June 15.
- [ ] Read all four research notes (`docs/research-notes/`).
- [ ] Each member runs `docs/proposals/03-data-pipeline.md` bootstrap to get a working dev env.
- [ ] Build minimal end-to-end demo on **LSC'24 public dataset** as a stand-in for the (not-yet-released) AIC dataset:
  - TransNetV2 -> SigLIP-2 -> FAISS flat -> simple Flask UI -> DRES local server.
- [ ] Internal mock-task practice - at least 3 sessions before June 15.
- [ ] **Decision gate**: confirm all licenses (PhoWhisper CC-BY-NC, jina-v3 CC-BY-NC) are acceptable to the organisers. *Verify in writing with the organising committee.*

### Phase 1 - Real-data baseline (June 25 -> mid-July) - 3 weeks
- [ ] AIC2026 preliminary content released June 25 - read & summarise the rules within 24 hours.
- [ ] Reproduce SigLIP-2 + Meta CLIP 2 baseline on the AIC dataset.
- [ ] Build the Milvus + Elasticsearch indexes for OCR/ASR.
- [ ] Build the React UI v1 with grid + temporal scrubber + filter sidebar.
- [ ] Wire DRES (or whatever scoring server organisers use).
- [ ] **Decision gate**: baseline must hit at least ~60% of estimated maximum on internal practice tasks.

### Phase 2 - Full system + fine-tuning (mid-July -> mid-Aug) - 4 weeks
- [ ] DreamLIP-style synthetic-caption generation with Qwen2.5-VL-72B INT4 -> LoRA fine-tune SigLIP-2 on synthetic Vi pairs.
- [ ] Train Vietnamese ColVintern late-interaction head on competition data.
- [ ] Add VLM-as-judge reranker (Vintern-3B-beta).
- [ ] Add DANTE DP for TRAKE.
- [ ] Add LLM planner (SeaLLMs-v3-7B) for both tracks.
- [ ] Add submission-verification panel (anti-foot-gun for the operator).
- [ ] **Decision gate**: full system beats baseline by 20%+ on internal practice.

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

**Development**: 1xRTX 4090 (24 GB) or 1xA6000 (48 GB) per engineer. Cloud burst to H100/H200 only for synthetic-caption generation and one-off fine-tunes.

**Finals**: bring 2 identical laptops (mirror images), redundant power, 4G hotspot backup. Pre-stage all model weights on each laptop (Vintern, PhoWhisper, SigLIP-2 quantised, Meta CLIP 2 quantised - total ~30 GB).

## 7. The eight things that will lose this competition if we skip them

1. **Submission-verification panel.** Every mistake costs 10 points. PraK1 vs PraK2 differed by 30 points on the same engine.
2. **TRAKE DP.** Pure semantic search scatters the correct 4-scene sequence across non-adjacent frames. Without DANTE-style DP we lose TRAKE points.
3. **Vietnamese ASR transcript cleanup.** Whisper-large-v3 makes diacritic errors that break proper-noun queries. Either use PhoWhisper or post-process with Gemini Flash.
4. **OOK entity handling.** Frozen CLIP can't embed local Vietnamese celebrities/brands. LLM query rewrite + external image search is the mitigation.
5. **Operator practice.** Allocate >=20% of total prep time to mock runs.
6. **Multi-model ensemble.** Single-model systems plateaued in 2023; the 2025 LSC winner attributed its win to swapping a single component (graph-DB -> Milvus). Build the ensemble from day 1.
7. **Verification of license compatibility.** PhoWhisper is CC-BY-NC. Confirm with organisers in writing before depending on it.
8. **Internal evaluation harness.** Don't measure progress by feelings. Write a fixed test set of mock queries and run nightly regression.

## 8. The five things we will deliberately *not* do

1. **VR/AR/eye-tracking.** Cool but adds operator cognitive load and hardware risk. Skip CollaXRSearch / EAGLE / vitrivr-VR style features.
2. **Knowledge graphs.** LifeGraph is academically interesting but slow and brittle. Use Milvus + structured fields instead.
3. **Custom model training from scratch.** All fine-tuning is LoRA + synthetic caption data; never train a backbone from scratch.
4. **More than 3 embedding models in the ensemble.** Diminishing returns; harder to debug.
5. **Pure-LLM end-to-end ("just throw it at GPT-4o").** LSC review SS IV-D: top QA systems were *not* purely conversational LLM. RAG works; chat-only does not.

## 9. Reading list (in priority order)

1. `docs/strategy/00-master-strategy.md` (this document)
2. `docs/research-notes/01-aic-hcmc-prior-editions.md` - what worked at this competition
3. `docs/research-notes/02-lsc-vbs-systems-deep-dive.md` - the world reference
4. `docs/proposals/01-interactive-system-architecture.md` - what we're building
5. `docs/proposals/02-automatic-track-agent.md` - the agent track
6. `docs/research-notes/03-foundation-models-2026.md` - model choices
7. `docs/research-notes/04-vietnamese-stack-and-agents.md` - Vietnamese specifics
8. `docs/proposals/03-data-pipeline.md`, `04-fine-tuning-plan.md`, `05-evaluation-harness.md`, `06-ui-ux-design.md`
9. The LSC SOTA review PDF in `docs/` (already there)
10. The downloaded papers in `docs/papers/` (skim abstracts; deep-read only when implementing)

## 10. Open questions / decisions to make in the next 2 weeks

1. **Team registration**: register through <https://263.org.vn/AIC2026-Registration> before June 15.
2. **License confirmation**: email/Slack the organising committee to confirm whether CC-BY-NC weights (PhoWhisper, jina-v3) are allowed.
3. **Hardware**: decide laptop vs desktop for finals; nail down GPU spec.
4. **Cloud budget**: how much will we spend on Gemini 2.5 Flash + GPT-4o during practice and finals?
5. **Dataset preview**: any way to access a sample of the AIC2026 data ahead of June 25? Ask HCMUS contacts.
6. **TRAKE in 2026?**: not confirmed yet that TRAKE remains a task in 2026; design for it but keep the rest task-agnostic.
7. **Operator candidates**: identify the 2 operators by June 1 and start their practice schedule.
