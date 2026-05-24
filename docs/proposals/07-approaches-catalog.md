# Proposal 07 - Catalog of All Feasible Approaches (with verdicts)

> The user asked us to "propose all feasible approaches". This document enumerates every approach worth considering, including ones we will NOT use, with rationale. Useful for sanity-checking our chosen path and for picking a fallback if our primary path fails.

## How to read this catalog

Each approach is rated on three dimensions:
- **Quality** (1-5): expected effectiveness on AIC2026-style tasks
- **Effort** (1-5): engineering hours to ship
- **Risk** (1-5): probability of failing badly

Our choice is marked **CHOSEN**. Strong alternatives marked **BACKUP**.

---

## A. Retrieval architecture

### A1. Single-model CLIP + FAISS flat
- Q: 2, E: 1, R: 1
- Verdict: **TOO WEAK** - plateaued in 2023 LSC; everyone does this; not enough to differentiate.

### A2. Single SOTA CLIP + Milvus HNSW
- Q: 3, E: 2, R: 1
- Verdict: **BASELINE** - good first cut. We start here in Phase 0.

### A3. **3-model ensemble (SigLIP-2 + Meta CLIP 2 + InternVideo2) + Milvus + RRF**
- Q: 5, E: 3, R: 2
- Verdict: **CHOSEN** - what every LSC/VBS top-3 does since 2022.

### A4. 5-model ensemble + late STR-Lucene fusion (VISIONE-style)
- Q: 5, E: 5, R: 3
- Verdict: **DIMINISHING RETURNS** - extra cost not justified by extra recall. VISIONE proves it's possible but they have years of engineering invested.

### A5. Knowledge graph + SPARQL (LifeGraph-style)
- Q: 3, E: 5, R: 4
- Verdict: **SKIP** - academically interesting but slow + brittle. LifeGraph teams have plateaued mid-rank.

### A6. Pure end-to-end VLM ("just throw at GPT-4o")
- Q: 2, E: 1, R: 5
- Verdict: **SKIP** - LSC review SS IV-D quantifies the failure; RAG works, chat alone does not.

### A7. ColPali / ColVintern multi-vector page retrieval (no dual-encoder)
- Q: 4, E: 3, R: 3
- Verdict: **BACKUP / ADDITIVE** - excellent for OCR-heavy frames. We will add ColVintern *alongside* the ensemble.

---

## B. Text encoder strategy for Vietnamese queries

### B1. English-only CLIP + machine-translate Vi->En
- Q: 2, E: 1, R: 3
- Verdict: **TOO LOSSY** - translation introduces errors for named entities.

### B2. Multilingual CLIP (M-CLIP)
- Q: 3, E: 1, R: 1
- Verdict: **BASELINE FALLBACK** - usable but outclassed by Meta CLIP 2.

### B3. **Meta CLIP 2 (multilingual from scratch)**
- Q: 5, E: 1, R: 1
- Verdict: **CHOSEN** - best XM3600 multilingual in 2025. MIT.

### B4. Train our own Vietnamese CLIP from scratch
- Q: ?, E: 5, R: 5
- Verdict: **NEVER** - way out of timeline.

### B5. LoRA-fine-tune SigLIP-2 on Vietnamese synthetic captions (DreamLIP)
- Q: 4, E: 3, R: 2
- Verdict: **CHOSEN AS ADD-ON** - free recall boost on top of B3.

---

## C. Vector index

### C1. FAISS flat
- Q: 5 (exact), E: 1, R: 1, but Latency: high
- Verdict: **DEV ONLY** - too slow at 1M+ scale.

### C2. **FAISS HNSW (M=32)** in-process
- Q: 4.5, E: 2, R: 1
- Verdict: **VIABLE** - simple, fast. Use if Milvus operationally complex.

### C3. **Milvus 2.5 with hybrid + structured fields**
- Q: 5, E: 3, R: 2
- Verdict: **CHOSEN** - MEMORIA (LSC'25) reported their win-attribution as the FAISS->Milvus swap; we treat that as motivating evidence, not a controlled ablation. Hybrid dense+filter in one query is the engineering win.

### C4. Qdrant
- Q: 5, E: 3, R: 2
- Verdict: **VIABLE ALTERNATIVE** - equally capable; team preference.

### C5. Weaviate
- Q: 4, E: 3, R: 3
- Verdict: **SKIP** - good but ecosystem matters; Milvus has more Vietnamese-team prior art.

### C6. ScaNN (Google)
- Q: 5, E: 4, R: 3
- Verdict: **SKIP** - C++ heavy, less Python-friendly tooling.

### C7. Vespa
- Q: 5, E: 5, R: 4
- Verdict: **SKIP** - powerful but overkill; high learning curve.

---

## D. Reranking

### D1. No rerank
- Q: 2, E: 0, R: 1
- Verdict: **SKIP** - leaves points on the table.

### D2. BLIP-2 ITM head
- Q: 3, E: 2, R: 1
- Verdict: **BACKUP** - classic; reliable; English-leaning.

### D3. BERT cross-encoder fine-tuned on local data
- Q: 4, E: 4, R: 3
- Verdict: **VIABLE** - MemoriEase 3.0 uses this for QA.

### D4. **VLM-as-judge (Vintern-3B-beta + Gemini 2.5 Flash escalation)**
- Q: 5, E: 3, R: 2
- Verdict: **CHOSEN** - 2026 default; Vietnamese-aware; CoT-capable.

### D5. ColBERT-style MaxSim
- Q: 4, E: 3, R: 2
- Verdict: **VIABLE** - via BGE-M3's ColBERT head; basically free if we already use BGE-M3.

### D6. LLM ranker via logit probing (InternVL 3.5 yes/no)
- Q: 4, E: 2, R: 2
- Verdict: **BACKUP** - cheaper than D4 but English-leaning.

---

## E. Query understanding

### E1. Pass query directly to dual encoder
- Q: 2, E: 0, R: 1
- Verdict: **SKIP** - loses to query-expansion methods.

### E2. **LLM query rewrite + paraphrase + intent JSON (SeaLLMs-v3 / Gemini Flash)**
- Q: 5, E: 3, R: 2
- Verdict: **CHOSEN** - NII-UIT VBS'25 winner used this; ViewsInsight 2.0 did too.

### E3. SDXL "generative visual query" (text -> image -> image search)
- Q: 4, E: 3, R: 3
- Verdict: **CHOSEN AS SPECIAL TOOL** - NII-UIT's unique trick; valuable for descriptive queries.

### E4. Sketch / canvas object placement (VISIONE)
- Q: 3, E: 4, R: 3
- Verdict: **SKIP** - cool but adds operator complexity; only marginally helpful.

### E5. External search (Google Lens / Bing Visual)
- Q: 3, E: 2, R: 3
- Verdict: **BACKUP** - last-resort for OOK entities; rate-limited.

---

## F. Vietnamese ASR

### F1. Whisper-large-v3
- Q: 3 (Vietnamese), E: 1, R: 1
- Verdict: **BACKUP** - decent fallback; MIT licensed.

### F2. **PhoWhisper-large**
- Q: 5 (Vietnamese SOTA, WER 8.14), E: 1, R: 2 (license)
- Verdict: **CHOSEN** - subject to license verification with organisers.

### F3. wav2vec2-large-vi-vlsp2020
- Q: 3, E: 2, R: 2
- Verdict: **SKIP** - beaten by PhoWhisper-medium.

### F4. Whisper-large-v3 + Gemini diacritic correction
- Q: 4, E: 2, R: 2
- Verdict: **VIABLE COMBO** - MERVIN approach; works if PhoWhisper not allowed.

### F5. Canary-1B-v2
- Q: 1 (Vietnamese not supported), E: 1, R: 1
- Verdict: **SKIP** for Vietnamese - viable for English code-switch only.

---

## G. Vietnamese OCR

### G1. **PaddleOCR PP-OCRv5** (Vietnamese latin recognizer)
- Q: 5, E: 1, R: 1
- Verdict: **CHOSEN** - SOTA, Apache-2.0.

### G2. **VietOCR fallback** for low-confidence boxes
- Q: 4, E: 2, R: 1
- Verdict: **CHOSEN** - SOTA Vietnamese-specific.

### G3. EasyOCR
- Q: 3, E: 1, R: 1
- Verdict: **BACKUP** - easiest install; lower quality.

### G4. Tesseract
- Q: 2, E: 1, R: 1
- Verdict: **SKIP** - legacy.

### G5. VLM-based OCR (Qwen2.5-VL-72B-INT4)
- Q: 4, E: 3, R: 2
- Verdict: **TARGETED USE** - only on the hardest 1% of frames where PP-OCRv5 fails.

---

## H. Vietnamese text retrieval (over OCR/ASR/captions)

### H1. BM25 with pyvi tokenizer
- Q: 3, E: 1, R: 1
- Verdict: **CHOSEN AS PART OF HYBRID** - essential for proper nouns.

### H2. **BGE-M3 dense + sparse + ColBERT (all in one)**
- Q: 5, E: 2, R: 1
- Verdict: **CHOSEN** - one model, three retrieval modes; MIT.

### H3. multilingual-e5-large-instruct
- Q: 4, E: 1, R: 1
- Verdict: **BACKUP** - best sub-1B on MMTEB.

### H4. PhoBERT-large fine-tuned
- Q: 3, E: 3, R: 2
- Verdict: **SKIP** - BGE-M3 covers it.

### H5. dangvantuan/vietnamese-embedding
- Q: 4, E: 1, R: 1
- Verdict: **BACKUP** - MERVIN's choice; smaller and faster.

### H6. SPLADE-v3 (English; train Vi version)
- Q: 4, E: 4, R: 3
- Verdict: **SKIP** - effort not justified given BGE-M3 sparse.

---

## I. TRAKE / temporal alignment

### I1. Independent retrieval per scene, no temporal constraint
- Q: 2, E: 1, R: 1
- Verdict: **TOO WEAK** - scatters frames; loses TRAKE points.

### I2. **DANTE-style dynamic programming with lambda penalty**
- Q: 5, E: 3, R: 2
- Verdict: **CHOSEN** - AIO_Owlgorithms' 2025 paper proves this wins.

### I3. Sliding-window heuristic
- Q: 3, E: 2, R: 2
- Verdict: **BACKUP** - simpler but less optimal.

### I4. Joint 4-frame embedding (clip-level retrieval)
- Q: ?, E: 5, R: 4
- Verdict: **NEVER** - architecturally unclear how to define.

---

## J. UI / UX

### J1. Minimal terminal CLI
- Q: 1, E: 1, R: 2
- Verdict: **SKIP** - operator speed too low.

### J2. Vanilla HTML form
- Q: 2, E: 1, R: 1
- Verdict: **SKIP** - no keyboard ergonomics.

### J3. **React + keyboard-everything + scrubber + verification bar**
- Q: 5, E: 4, R: 2
- Verdict: **CHOSEN**.

### J4. VR / AR interface
- Q: 3, E: 5, R: 5
- Verdict: **SKIP** - hardware + cognitive risk; not differentiating at our level.

### J5. Eye-tracking integration (EAGLE)
- Q: 3, E: 5, R: 5
- Verdict: **SKIP** - same reasons.

### J6. Mobile-friendly (MyEachtraX style)
- Q: 2, E: 3, R: 2
- Verdict: **SKIP** - desktop is the competition format.

---

## K. LLM for planner / agent

### K1. **SeaLLMs-v3-7B function-calling (local)** for routine queries
- Q: 4, E: 3, R: 2
- Verdict: **CHOSEN PRIMARY**.

### K2. **Gemini 2.5 Flash** for hard queries (escalation)
- Q: 5, E: 1, R: 2 (API outage)
- Verdict: **CHOSEN ESCALATION**.

### K3. GPT-4o / o4-mini
- Q: 5, E: 1, R: 2
- Verdict: **BACKUP** if Gemini outage.

### K4. Vistral-7B function-calling
- Q: 4, E: 3, R: 2
- Verdict: **BACKUP** - Vietnamese-native but smaller community.

### K5. Llama-3.x Vietnamese fine-tune
- Q: 3, E: 3, R: 3
- Verdict: **SKIP** - quality varies.

### K6. PhoGPT
- Q: 3, E: 2, R: 3
- Verdict: **SKIP** - older, smaller community.

---

## L. Fine-tuning strategy

### L1. **LoRA on SigLIP-2 with DreamLIP-style synthetic captions**
- Q: 4, E: 3, R: 2
- Verdict: **CHOSEN**.

### L2. **LoRA on ColVintern for OCR-heavy pages**
- Q: 4, E: 3, R: 2
- Verdict: **CHOSEN**.

### L3. Cross-encoder Vintern-1B fine-tune
- Q: 4, E: 3, R: 2
- Verdict: **CHOSEN**.

### L4. Distillation big -> small CLIP
- Q: 3, E: 4, R: 3
- Verdict: **SKIP** - not necessary at our scale.

### L5. Full fine-tune from scratch
- Q: ?, E: 5, R: 5
- Verdict: **NEVER**.

---

## M. Agent framework for automatic track

### M1. **LangGraph state machine**
- Q: 5, E: 3, R: 2
- Verdict: **CHOSEN**.

### M2. DSPy
- Q: 4, E: 3, R: 2
- Verdict: **CHOSEN AS PROMPT OPTIMISER** - layered on top of LangGraph.

### M3. smolagents
- Q: 3, E: 2, R: 3
- Verdict: **BACKUP** - simpler but less mature.

### M4. PydanticAI
- Q: 4, E: 2, R: 3
- Verdict: **BACKUP** - good type safety.

### M5. OpenAI Agents SDK
- Q: 4, E: 2, R: 3
- Verdict: **SKIP** - OpenAI-centric; locks us in.

### M6. AutoGen / Magentic-One
- Q: 3, E: 5, R: 4
- Verdict: **SKIP** - heavy multi-agent overkill.

### M7. Hand-rolled state machine (Python only)
- Q: 4, E: 4, R: 2
- Verdict: **BACKUP** - if LangGraph deps break.

---

## N. Operator training

### N1. No formal training
- Q: 1, E: 0, R: 5
- Verdict: **NEVER** - PraK1 vs PraK2.

### N2. Ad-hoc practice sessions
- Q: 3, E: 1, R: 3
- Verdict: **MINIMUM ACCEPTABLE**.

### N3. **Structured operator drills + recorded sessions + weekly speedruns**
- Q: 5, E: 3, R: 1
- Verdict: **CHOSEN**.

### N4. Multiple system instances with multiple operators
- Q: 3, E: 4, R: 4
- Verdict: **SKIP** - LSC review SS IV-D: more variance, not more score.

---

---

## O. Original contributions on top of the SOTA stack

Sections A-N describe the *reused* SOTA stack. Sections O1-O5 describe what we add on top: our original contributions. Full method, eval, and risk discussion in [`08-original-contributions.md`](08-original-contributions.md).

### O1. **DiacriticBERT - Vietnamese diacritic-robust late-interaction head** (C1)
- Q: 4, E: 2, R: 2
- Verdict: **CHOSEN (primary)** - first retrieval head explicitly trained on a controlled Vietnamese diacritic-noise schedule. Targets a failure mode (item 3 in master strategy SS 7) that off-the-shelf BGE-M3 ignores. ~1 week, ~3 GPU-hours. Fallback if it fails ablation: SeaLLMs-v3 query-rewriting at index time.

### O2. **Per-task-type learned fusion** (C2)
- Q: 4, E: 1, R: 1
- Verdict: **CHOSEN (primary)** - replaces uniform RRF k=60 (Cormack 2009) with a LightGBM LambdaRank model selected at query time by the planner's emitted `task_type`. ~3 days, ~30 minutes CPU. Runtime auto-fallback to RRF if the learned model regresses on a streaming 50-query window makes this risk-free in production.

### O3. **PriorDP - story-graph generalisation of DANTE for TRAKE** (C3, backup)
- Q: 4, E: 4, R: 3
- Verdict: **BACKUP** - generalises DANTE's linear lambda penalty with a learned scene-transition prior matrix. Ships only if Phase 2 has slack AND TRAKE remains a task type in 2026 (not confirmed). ~2 weeks.

### O4. **Agent self-distillation via DSPy on operator traces** (C4)
- Q: 5, E: 2, R: 2
- Verdict: **CHOSEN (primary)** - the interactive operator's correct, fast submissions are the training corpus for the automatic-track planner via DSPy MIPRO. Pattern only exists because 2026 is the first year the automatic track is a serious sub-event. ~1 week after Phase 1 instrumentation; refreshes continually through prelims and mock-finals.

### O5. **Counterfactual VLM rerank for OOK named entities** (C5, backup)
- Q: 4, E: 2, R: 2
- Verdict: **BACKUP** - replaces direct "rank these 9" prompts with iterative counterfactual pruning + 3-vote shuffle, attacking the documented position-bias of VLM judges. ~1 week. Ships only if dev-set ablation shows >=5% R@1 on the long-tail-entity slice.

---

## Summary

Our chosen approach is **A3 + B3 + B5 + C3 + D4 + E2 + E3 + F2 + G1+G2 + H1+H2 + I2 + J3 + K1+K2 + L1+L2+L3 + M1+M2 + N3 + O1+O2+O4**. The A-N items are the reused 2026 SOTA stack (aligns with empirical evidence from LSC'22-25, VBS'22-25, AIC HCMC'23-25 winners). The O1+O2+O4 items are our three primary original contributions. O3 and O5 are backups conditional on Phase 2 slack. This is the highest-expected-value combination of "reproduce the floor + add a defensible edge" given a 17-week timeline and a 5-person team.
