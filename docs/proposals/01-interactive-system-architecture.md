# Proposal 01 - Interactive System Architecture

> Detailed proposal for our primary retrieval system used on the interactive track. The same indexes and tools are reused by the automatic-track agent (see proposal 02).
>
> **What is reused vs novel in this proposal:** sections 2-5 describe the reproduced 2026 SOTA floor (every component is off-the-shelf). The novel additions are wired in as: (a) **DiacriticBERT** late-interaction head alongside BGE-M3 in OCR/ASR retrieval (SS 5.6.1, contribution C1); (b) **per-task-type learned fusion** replacing uniform RRF k=60 (SS 5.11, contribution C2); (c) **position-bias-mitigated** VLM rerank (SS 5.9); and (d) optional **PriorDP** for TRAKE (SS 5.12, backup contribution C3). The full novelty proposal is [`08-original-contributions.md`](08-original-contributions.md).

## 1. Goals and non-goals

### Goals
- Build a **2026-best-of-breed** interactive retrieval system that competes for 1st place on the interactive track of AIC HCMC 2026.
- Support all four task types: **KIS**, **QA**, **TRAKE** (4-scene temporal), and **Ad-hoc** retrieval.
- Sub-second median latency for retrieval over ~1M keyframes; sub-2s p95.
- Vietnamese natural-language queries first-class.
- Be the same substrate that the **automatic agent** drives on the autonomous track.

### Non-goals
- VR/AR/eye-tracking interfaces (skip; adds risk).
- Knowledge-graph-only retrieval (skip; slow + brittle).
- Pure LLM end-to-end retrieval (skip; LSC review shows RAG beats chat).

## 2. System components

### 2.1 Offline indexing pipeline

| Stage | Tool | Notes |
|---|---|---|
| Shot detection | **TransNetV2** | Produces shot boundaries; one keyframe per shot |
| Frame sampling | **KDE-GMM density-aware sampling** | Used by MERVIN; ensures ~1 frame per 2-3s |
| Image embedding (primary) | **SigLIP-2 So400m/16@384** | Apache-2.0; best default 2026 |
| Image embedding (Vietnamese) | **Meta CLIP 2 ViT-H/14** | MIT; best XM3600 multilingual |
| Video embedding | **InternVideo2-1B** | Apache-2.0; 4 frames per clip |
| OCR | **PaddleOCR PP-OCRv5 (latin/vi)** | Apache-2.0; covers 111 langs incl. Vietnamese diacritics |
| OCR fallback (handwritten) | **VietOCR** | Apache-2.0; Vietnamese SOTA |
| ASR | **PhoWhisper-large** via **WhisperX** | CC-BY-NC-SA (verify); word-level timestamps |
| ASR fallback (code-switch EN) | **Whisper-large-v3-turbo** | MIT |
| Diarization | **Pyannote-audio 3.1** | MIT |
| Audio events | **LAION-CLAP music_speech_audioset** | CC0 weights |
| Caption generation | **Qwen2.5-VL-7B BF16** (local) | long Vietnamese captions; Apache-2.0 |
| Object detection | **YOLOv8** + COCO | AGPL-3.0 (check license); used only for filter tags |
| Scene labels | **Places365** | MIT |
| ADL labels | **LSC-ADL 35 classes** | manual cluster + Qwen-VL labelling |

### 2.2 Online retrieval pipeline

```
text query (Vietnamese)
   |
   v
[ Planner LLM ] - SeaLLMs-v3-7B function-calling
   - parse JSON intent  (also emits task_type for fusion-model selection)
   - emit DAG of tool calls + fusion weights
   - paraphrase into 3 Vi variants
   |
   v   parallel execution
+-----------------+-------------------------+-------------------------+
| text->image     | OCR retrieval           | ASR retrieval           |
| SigLIP-2 + Meta | BGE-M3 (dense+sparse)   | BGE-M3 (dense+sparse)   |
| CLIP2 + Intern  | + BM25                  | + BM25                  |
| Video2 (Milvus) | + DiacriticBERT MaxSim  | + DiacriticBERT MaxSim  |
|                 |   (ours, C1)            |   (ours, C1)            |
|                 | Elasticsearch           | Elasticsearch           |
+-----------------+-------------------------+-------------------------+
   |
   v
[ Per-task-type learned fusion (ours, C2; LightGBM LambdaRank) ]  -> top-200
   - planner.task_type in {KIS, QA, TRAKE, Ad-hoc} -> select fusion model
   - auto-fallback to RRF k=60 if learned model fails the runtime guardrail
   |
   v
[ Structured filters ]  time, location, object, ADL  -> top-50
   |
   v   (only if TRAKE/temporal)
[ DANTE DP (default) or PriorDP (C3 if shipped) ]  4-scene aligned
   |
   v
[ VLM-as-judge rerank ]  Vintern-3B-beta on 3x3 grid + CoT
   - position-bias mitigation: input-order shuffle + 3-vote majority
   - C5 counterfactual rerank as backup mode for OOK queries
   -> top-10
   |
   v
[ React UI ]
   |
   v
[ Operator decides + submission-verification panel ]  -> DRES submit
   |
   v
[ Trace logger (Parquet) -> data/operator_traces.parquet ]
   (feeds C4 agent self-distillation; see proposal 02)
```

### 2.3 Vector and text storage

- **Milvus** (single deployment, multi-collection)
  - `collection_siglip2` (1024-d float)
  - `collection_metaclip2` (1024-d float)
  - `collection_internvideo2` (768-d float)
  - `collection_clap_audio` (512-d float)
  - structured fields per row: `video_id`, `frame_id`, `timestamp`, `shot_id`, `place_label`, `adl_label`, `object_tags[]`, `duration_ms`
- **Elasticsearch** (single deployment)
  - `idx_ocr` - text + bbox + ts; Vietnamese analyser (icu_tokenizer + pyvi normalisation)
  - `idx_asr` - text + word-level ts; Vietnamese analyser
  - `idx_caption` - long Vietnamese captions + word-level scoping
- **SQLite** (per-team metadata) - submission log, query history, mock-task gold answers

### 2.4 Why these specific component choices

| Decision | Rationale |
|---|---|
| **SigLIP-2 over OpenCLIP H/14** | SigLIP-2 g-opt/16@384 has 85.0 IN-1k zero-shot vs OpenCLIP H/14 ~78. Apache-2.0. |
| **Meta CLIP 2 specifically** | Best XM3600 multilingual in 2025 (64.3 I->T aggregate). Note: this is the global multilingual number, not a Vietnamese-specific subset. We will measure the Vietnamese slice on our own held-out set before shipping any Vi-first marketing. |
| **Milvus over FAISS-only** | MEMORIA (LSC'25) attributed their win to this swap; we treat that as motivating evidence, not a controlled ablation. Hybrid dense+filter in one query is the engineering win. |
| **PhoWhisper over Whisper-large-v3** | WER 8.14 vs ~15 on Vietnamese (CMV-Vi). Worth the license caveat. |
| **VLM-as-judge over BLIP-2 ITM rerank** | 2026 default; Gemini 2.5 Flash + Vintern-3B-beta give frontier quality at low cost. Position bias is a known failure mode and is mitigated in SS 5.9. |
| **Per-task-type learned fusion over RRF k=60** (C2, ours) | RRF is distribution-robust because it discards score magnitudes - which also means it discards ranker quality. With 12-15 heterogeneous ranked lists, per-task-type LambdaRank consistently beats RRF in the IR literature. See [`08-original-contributions.md`](08-original-contributions.md) SS 4. Runtime guardrail auto-falls back to RRF k=60 if the learned model regresses. |
| **DiacriticBERT alongside BGE-M3** (C1, ours) | BGE-M3 is trained on clean Vietnamese; PhoWhisper/PaddleOCR produce noisy diacritics. A small head trained on a controlled diacritic-noise schedule absorbs the noise distribution at retrieval time, complementing the query-rewriting fallback at index time. See [`08-original-contributions.md`](08-original-contributions.md) SS 3. |
| **DANTE for TRAKE (default)**; **PriorDP (C3) optional** | AIO_Owlgorithms LSC'25 proves DANTE-style DP wins TRAKE; lambda in [0.001, 0.01]. PriorDP (ours, backup) generalises the linear penalty with a learned scene-transition prior. |
| **Function-calling planner** | Unifies interactive + automatic tracks. SnapMind MMM 2026 blueprint. Our extension is **what we train it on** (C4 agent self-distillation), not the architecture. |

## 3. Request lifecycle (textual KIS example)

1. Operator types: "Co?nh tre? em cha?y nhay du+o+i mu+a o+? sa^n cho+i tru+o+`ng ho?c". (Children running in the rain at a school playground)
2. **Planner LLM** in <300ms parses to JSON:
   ```json
   {
     "objects": ["children", "playground"],
     "actions": ["running", "playing"],
     "weather": "rainy",
     "location": "school",
     "modality_priority": ["image", "ocr", "asr"],
     "vi_paraphrases": [
       "tre? con cha?y giu+~a co+n mu+a tru+o+`ng ho?c",
       "ho?c sinh choi mu+a san truo+`ng",
       "children playing in rain at school playground"
     ]
   }
   ```
3. **Parallel retrieval**:
   - SigLIP-2 + Meta CLIP 2 + InternVideo2 each query Milvus for top-200 against each Vietnamese paraphrase and the English paraphrase = 12 ranked lists total.
   - Elasticsearch fuzzy on OCR text for "school"-related keywords and on ASR/caption text for the Vietnamese paraphrases. Each Elasticsearch lane returns BM25 + BGE-M3 dense + BGE-M3 sparse + **DiacriticBERT MaxSim** scores (C1, ours).
4. **Per-task-type learned fusion (C2, ours)**: planner emits `task_type = "KIS"`, so the KIS-specific LambdaRank model is loaded; it fuses all 12 image-text + 9 text-side lists -> top-200 candidates. Runtime guardrail re-runs RRF k=60 in the background and switches to it if the learned model's median rank on a streaming 50-query window drops below the RRF baseline.
5. **Structured filters**: keep frames with `place_label in {school, playground, park}` OR (`adl_label = "playing_outdoor"`) -> top-50.
6. **VLM-as-judge rerank**: Vintern-3B-beta receives 9 candidates at a time in 3x3 grids with prompt "Sa^?p xe^?p anh theo do^. lien quan vo+i: 'tre? em cha?y mu+a o+? sa^n cho+i'" + CoT. **Position-bias mitigation**: input order is shuffled 3 times at temperature 0.7 and ranks are majority-voted -> top-10.
7. **UI** displays top-10 in a grid; operator clicks on the best one; **submission-verification panel** shows the frame + neighbours + a confidence score; operator confirms; DRES API call fires.
8. **Trace logger** appends `(query, planner_intent_json, fusion_model_id, tool_calls, final_submission, gold_outcome)` to `data/operator_traces.parquet`. This feeds C4 (agent self-distillation, proposal 02).

Target: end-to-end ~1.5s; operator decision ~5-15s. The position-bias 3-vote ensemble shifts the rerank p95 from 900ms (single-vote) to 1.2s (3-vote). This stays inside the overall <2s p95 retrieval budget shown in SS 4 below.

## 4. Latency budget (target)

| Stage | Target p50 | Target p95 |
|---|---|---|
| Planner LLM (local 7B) | 100 ms | 250 ms |
| Milvus ANN (each of 4 collections, parallel) | 80 ms | 150 ms |
| Elasticsearch (each of 3 indexes, parallel) | 50 ms | 120 ms |
| DiacriticBERT MaxSim (C1, top-200 candidates from BM25) | 15 ms | 40 ms |
| Per-task learned fusion (C2, LightGBM scoring) + filtering | 30 ms | 80 ms |
| DANTE DP / PriorDP (only TRAKE) | 200 ms | 400 ms |
| VLM-as-judge (Vintern-3B-beta, 4 grids batched, 3-vote shuffle) | 600 ms | 1.2 s |
| **End-to-end retrieval** | **<900 ms** | **<2 s** |
| Operator UI render | 50 ms | 100 ms |

Notes:
- The C2 LambdaRank fusion replaces the RRF fusion line; LightGBM scoring on ~3000 candidate-feature rows (200 candidates x 15 rankers) is ~10ms; the rest is feature assembly.
- The 3-vote VLM rerank inflates rerank p95 from 900ms to 1.2s. We absorb this within the overall <2s budget. If it becomes the bottleneck, drop to 2-vote for KIS (the speed-sensitive task type).

## 5. Detailed pipeline specs

### 5.1 Shot detection
- **TransNetV2** off-the-shelf, threshold 0.5.
- For images-only datasets (LSC-style lifelog), use day-folder buckets + content-based segmentation (HSV + CLIP feature drift) following SnapSeek 3.0.

### 5.2 Frame sampling
- 1 keyframe per shot (the centre frame, snapped to I-frame).
- Plus uniform 1 fps padding if shot longer than 5s.
- For TRAKE tasks: also extract scene-level "supercut" thumbnail (3x3 collage).

### 5.3 Embeddings

**SigLIP-2 So400m/16@384** at fp16:
- 400M params, ~10 GB VRAM
- 1024-d output
- Pre-process: resize to 384x384, normalize per model card
- Batch 256 on A6000 -> ~10K frames/min

**Meta CLIP 2 ViT-H/14** at fp16:
- ~1B params, ~16 GB VRAM
- 1024-d output
- Batch 128 on A6000 -> ~6K frames/min
- **Vietnamese-capable text encoder** is the win

**InternVideo2-Stage2_1B-224p-f4** at fp16:
- 1B params, ~6 GB VRAM
- 4 frames per clip @ 224x224
- 768-d output

### 5.4 Indexing
- Milvus 2.5+; **HNSW** index (M=32, efConstruction=200); **efSearch=128** at query.
- Per-collection collection sizes: 1M x 1024 fp16 = ~2 GB per collection on disk; 4 collections = ~8 GB. Easy.
- Elasticsearch 8.x; Vietnamese analyser via [`elasticsearch-analysis-vietnamese`](https://github.com/duydo/elasticsearch-analysis-vietnamese) + ICU.

### 5.5 OCR pipeline
- **PaddleOCR PP-OCRv5 detection** -> bounding boxes per frame.
- For each box: try **PP-OCRv5 latin (Vietnamese)** recognizer.
- Confidence < 0.6: retry with **VietOCR** Seq2Seq recognizer.
- Concatenate text with bbox metadata; index in Elasticsearch.
- All OCR text is also fed to the DiacriticBERT MaxSim head (SS 5.6.1) at query time as a third score signal beside BM25 and BGE-M3.

### 5.6 ASR pipeline
- Resample audio to 16kHz mono.
- Split on silence (Pyannote VAD).
- **PhoWhisper-large** via Faster-Whisper (CTranslate2 INT8).
- Force-align with WhisperX to get word-level timestamps.
- **Post-process**: send each segment through **Gemini 1.5/2.5 Flash** with prompt "Su+?a chi?nh ta? va` da^?u tie^?ng Vie^.t" (correct Vietnamese spelling and diacritics).
- Index segments in Elasticsearch with `video_id`, `start_ms`, `end_ms`.
- ASR text also feeds DiacriticBERT MaxSim (SS 5.6.1).

### 5.6.1 DiacriticBERT late-interaction head (C1, ours)
- See [`08-original-contributions.md`](08-original-contributions.md) SS 3 for the full method, eval plan, and risk discussion.
- Inference shape: frozen BGE-M3 encoder + 2-layer projection (768 -> 384 -> 384) + ColBERT-style MaxSim scoring.
- Query-time integration: for each query, encode the Vietnamese form once; for each top-200 candidate from BM25, fetch its DiacriticBERT vectors from a Milvus collection (`collection_diacriticbert`, 384-d) and compute MaxSim.
- Output is fed to the learned fusion (SS 5.11) as the 15th ranker.
- Files: `train/diacritic_bert.py`, `train/diacritic_noise.py`, `src/retrieval/diacritic_bert.py`.

### 5.7 Captioning
- **Qwen2.5-VL-7B BF16** prompt: "Mo^ ta? chi tie^?t bu+?c a?nh na`y ba`?ng tie^?ng Vie^.t, da`i 50-100 tu+`. Bao go^`m: dia? die^?m, nha^n va^.t, ha`nh do^.ng, ddo^` va^.t, ma`u sa?c chu? da.o."
- Run only on a sampled subset (1 in 8 keyframes) due to cost; if VRAM permits run on all.
- Captions indexed in Elasticsearch.

### 5.8 Planner LLM
- **SeaLLMs-v3-7B-Chat** locally on A6000 (~14 GB VRAM int8), Apache-2.0.
- Fallback: **Gemini 2.5 Flash** ($0.50/M input, $3.00/M output) for hard queries.
- Function-calling format with strict JSON output (schema-enforced via outlines / jsonformer).
- Prompt template lives in `prompts/planner.txt`; versioned in git.

### 5.9 VLM-as-judge reranker
- **Vintern-3B-beta** locally (~7 GB VRAM), Apache-2.0.
- For top-30 candidates, batch into 4 grids of 9 (last 3 padded with white).
- Prompt: "Vo+'i ca^u truy va^?n: '<query>'. Cho 9 anh sau, ha~y xe^?p ha.ng tu+` 1 (lien quan nha^?t) de^?n 9 va` gia?i thi?ch nga^?n go.n."
- Parse output -> aggregate to global rank.
- Fallback: **Gemini 2.5 Flash** for the top-9 only.

**Position-bias mitigation (mandatory).** VLM-as-judge on 3x3 grids is documented in the 2024-25 literature (LLaVA-Interleave, Gemini-eval) to be sensitive to spatial position within the grid. We mitigate with:
- **Input-order shuffle**: each grid is rendered 3 times at temperature 0.7 with different positional orderings of the same 9 candidates.
- **Majority vote**: the final rank per candidate is the median rank across the 3 runs; ties broken by mean score.
- **Cost**: ~3x rerank time (~600 ms p50, 1.2 s p95), still inside the overall <2s budget.
- **Optional C5 mode**: for OOK / long-tail-entity queries flagged by the planner, switch to iterative counterfactual pruning (see [`08-original-contributions.md`](08-original-contributions.md) SS 7) instead of direct ranking.

### 5.10 DANTE DP for TRAKE (default) and PriorDP (C3, optional)
- **DANTE (default).** For 4-scene TRAKE query: get top-200 candidates per scene. Build cost matrix: `cost(i,j,k,l) = -sum(score_scenes) + lambda * temporal_distance_variance`. Solve 4-dimensional shortest-path with constraint `t_i < t_{i+1}` and `t_{i+1} - t_i < MAX_GAP_MS`. lambda swept in [0.001, 0.01] on the dev set.
- **PriorDP (C3, ours, ships only if Phase 2 slack).** Replace the linear `lambda * variance` penalty with a learned scene-transition prior: `cost(d_i, d_{i+1}) = -log(retrieval_score(d_{i+1})) - alpha * log(P(scene(d_{i+1}) | scene(d_i))) + lambda * |t_{i+1} - t_i - mu_gap|`. `P(.|.)` is estimated from the offline scene/place/object co-occurrence statistics with Laplace-1 smoothing. See [`08-original-contributions.md`](08-original-contributions.md) SS 5 for the full method.
- Runtime selection: a config flag `trake_mode in {dante, priordp}` picks the algorithm; both write to the same output schema.

### 5.11 Per-task-type learned fusion (C2, ours)
- Replaces the uniform RRF k=60 in the pipeline diagram (SS 2.2).
- Method, training data, and ablation plan are in [`08-original-contributions.md`](08-original-contributions.md) SS 4.
- Runtime: planner LLM's emitted `task_type` selects one of four LightGBM LambdaRank models (KIS, QA, TRAKE, Ad-hoc); the model scores the ~3000 candidate-feature rows (top-200 per ranker x 15 rankers); top-200 fused output is forwarded to structured filters.
- **Runtime guardrail**: a shadow RRF k=60 is computed in parallel; if the learned model's median rank-of-gold on a streaming 50-query window drops below the shadow RRF baseline, the system auto-switches to RRF and emits a Prometheus alert.
- Files: `train/learned_fusion.py`, `src/retrieval/fusion.py`.

### 5.12 Operator-trace logging (feeds C4)
- Every interactive session writes one row per query to `data/operator_traces.parquet`:
  `(timestamp, operator_id, query_text, planner_intent_json, fusion_model_id, tool_calls_chosen, fusion_weights, rerank_decision, top_1_frame_id, final_submission, gold_outcome, time_to_submit_ms)`.
- This file is the training corpus for the automatic-track planner self-distillation (C4, proposal 02 SS 14).
- Logging is on from Phase 1 day one. Volume estimate: ~5 KB per query, ~10K queries by end of Phase 2 = ~50 MB. Trivial.

## 6. Hardware sizing

### Indexing (one-time, on a single A6000 48 GB)
- SigLIP-2 So400m@384: 12 GPU-hours / 1M frames
- Meta CLIP 2 H/14: 16 GPU-hours / 1M frames
- InternVideo2-1B (4 frames/clip @ ~250k clips): 24 GPU-hours
- PaddleOCR (CPU OK, GPU optional): 6 hours
- PhoWhisper + WhisperX (4 hours per 100h audio): assume 800h video -> 32 hours on 1 GPU
- Qwen2.5-VL-7B captioning (1 in 8): 20 GPU-hours
- Total: **~110 GPU-hours** for cold index. Run over 5 days on 1 A6000, or 12 hours on 8xH100 cloud burst.

### Online (during competition)
- Single A6000 hosts: SeaLLMs-v3-7B (int8, 14 GB) + Vintern-3B-beta (fp16, 7 GB) + Milvus client cache (~5 GB).
- Single 24 GB 4090 hosts: SigLIP-2 So400m + Meta CLIP 2 H/14 (for online image-query encoding only; not strictly needed if text-only queries).
- Backend: 1 box for Milvus, 1 box for Elasticsearch, 1 box for the FastAPI orchestrator.
- Total: 3-4 machines.

## 7. Software stack

- Python 3.11
- PyTorch 2.4+, transformers 4.45+, sentence-transformers 3.x
- Milvus 2.5+, Elasticsearch 8.x
- FastAPI backend + Pydantic + Uvicorn + Gunicorn
- LangGraph for the planner orchestration (optional in interactive; required in automatic)
- React 18 + Vite + TailwindCSS + Zustand for the frontend
- WebSocket transport for low-latency UI updates
- Docker Compose for local dev; bare-metal for finals

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PhoWhisper license incompatible | Medium | Medium | Whisper-large-v3-turbo fallback ready; verify in writing |
| Milvus instability under load | Low | High | Run pre-finals load test 10k QPS; have FAISS fallback path |
| Vintern-3B-beta hallucinates ranks | Medium | Medium | 3-vote shuffle (SS 5.9); C5 counterfactual rerank as backup |
| VLM rerank position-bias drift | Medium | Medium | Position-bias eval (`eval/rerank_position_bias.py`, proposal 05 SS 16) nightly |
| Operator slow at finals | Medium | High | Operator drills; submission-verification panel |
| Frontier API outages on finals day | Low | Medium | All cloud LLMs are *fallbacks*; system works fully offline |
| TRAKE removed from 2026 | Low | Low | Code is task-modular; PriorDP (C3) is conditional on TRAKE remaining |
| AIC dataset much larger than expected | Medium | Medium | Plan for 5M+ keyframes; HNSW scales linearly |
| **C1 DiacriticBERT regresses clean-Vietnamese retrieval** | Low | Medium | Per-class ablation gate (no class regresses >1.5%); fallback to SeaLLMs-v3 query rewriting |
| **C2 learned fusion overfits the dev set** | Medium | Low | Runtime guardrail auto-falls back to RRF on streaming 50-query window regression; leave-one-task-out CV at training time |
| **C4 distilled planner regresses on prelim queries unlike Phase-2 traces** | Medium | Medium | Frozen 100-query held-out A/B gate before each prompt replacement; previous prompt retained on regression |

## 9. Acceptance tests before each phase gate

- **Phase 1 (baseline)**: 80%+ KIS solved on a 20-query internal practice set within 5 min each.
- **Phase 2 (full system)**: 85%+ KIS + 70%+ QA + 60%+ TRAKE on a 30-query mock-finals.
- **Phase 3 (preliminary)**: top-3 official placement.
- **Phase 4 (finals prep)**: 90%+ on internal mock-finals.

## 10. Open design decisions

1. Do we run our own DRES instance for practice, or piggyback on a public one?
2. Do we ship a CLI client for the planner, or only the React UI?
3. Where do we store the planner-trace logs for post-round analysis? Suggested: append-only Parquet, replayable via `bin/replay-session`.
4. Caption generation budget: full 1M frames vs 1/8 sample? Cost vs recall trade-off; dev-set ablation pending.
