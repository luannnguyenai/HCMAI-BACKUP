# Research Note 05 ù Analysis of the 2025 baseline repo

> Permanent reference for the codebase delivered by last year's team. Use this when borrowing patterns under [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md), when writing a spec for a module with prior art, or when validating one of our Edge contributions.

**Repo**: <https://github.com/ThanhToan2111/AIC_2026.git>
**Local checkout**: `C:\Dev\AIC2026-SCOAI\AIC_2026` (parent dir of this workspace)
**Commit analysed**: `c3c3545` ù "Initial clean commit" (2026-05-26)
**Git history**: wiped to a single commit; we see only the final state of the 2025 deliverable, not its evolution.
**Author affiliation**: the original author (GitHub [`ThanhToan2111`](https://github.com/ThanhToan2111)) is a current member of the AIC2026 team (confirmed 2026-05-26). This changes the borrowing posture under [ADR-0010 ù3](../adr/ADR-0010-borrow-from-2025-baseline.md) ù explicit author permission is straightforward to obtain. The permission record + interview agenda is at [`docs/permissions/2025-baseline-reuse.md`](../permissions/2025-baseline-reuse.md). Several open questions in ù7 below become action items to resolve during that conversation rather than unknowns.

---

## 0. Executive summary

The 2025 baseline is a **functional Streamlit + FastAPI + Qdrant** system with a stack of 2024-era models (CLIP ViT-B/32 default, SigLIP v1, BEiT-3, OpenCLIP, BLIP-2, InternVL3) and cloud-only Vietnamese language handling (Whisper API, Gemini OCR, GPT-4o). It contains roughly **a week's worth of reusable plumbing** ù DRES integration, Pydantic request schemas, BLIP-2 reranker glue, TransNetV2 wrapper + weights, a primitive TRAKE algorithm ù that we should harvest under the policy in [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md). It also independently validates each of our Edge contributions:

- **C1 DiacriticBERT** ù their Elasticsearch analyser literally strips Vietnamese diacritics (asciifolding filter, `elasticsearch_service.py:32-43`). Their text retrieval over OCR/ASR was diacritic-blind. C1 attacks this exact failure mode.
- **C2 Per-task-type learned fusion** ù they had global RRF k=60 with optional per-query weight override (`api_server.py:58-59`); no per-task tuning, no LambdaRank.
- **C4 Agent self-distillation** ù they had no autonomous-track agent at all. The automatic track is greenfield for us.

The repo also corroborates every diagnosis from the May 25 meeting that led to [ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md) (RTX 5070), [ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md) (no Streamlit), [ADR-0005](../adr/ADR-0005-llm-path-bakeoff-gates-planner.md) (bakeoff), and [ADR-0006](../adr/ADR-0006-int4-quantization-hot-path.md) (quantization).

---

## 1. Repo at a glance

```
AIC_2026/
  README.md                        # 1.3 KB - bare bones; says "AI Challenge 2025"
  requirements.txt                 # 0.4 KB - 29 deps, mixed licenses
  config.py                        # 0.5 KB - CLIP-ViT-B/32 + GPT-4o defaults
  run_app.sh                       # 1.0 KB - venv + Qdrant docker + streamlit
  av1_to_h264.py                   # ffmpeg wrapper
  streamlit_api.py                 # 147.8 KB - MONOLITHIC UI (single file)
  api/
    api_server.py                  # 107.2 KB - FastAPI backend
  app/
    chatbot.py                     # LangChain single-shot RAG, GPT-4o
    elasticsearch_service.py       # 12.6 KB - ES wrapper (diacritic-stripping analyser)
    vector_store.py                # 1.6 KB - thin Qdrant wrapper
    vector_store_optimized.py      # 31.2 KB - production Qdrant wrapper
  model/                           # 20 files; every embedding + reranker tried
    Beit3_embedding_optimized.py
    BLIP2_rerank.py
    CLIP_embedding.py
    Gemini_OCR.py / GLM_OCR.py / MultiLLM_OCR.py    # CLOUD OCR
    GPT4o_service.py
    InternVL_rerank.py
    OpenCLIP_embedding{_optimized}.py
    Phi3_expand_query.py
    SigLIP_embedding.py
    SigLIP2_rerank{_optimized}.py
    StableDiffusion_genImg.py      # generative visual query
    transnetv2_pytorch.py
    Whisper_API.py
    Whisper_VAD.py
    YOLOE_object_detection.py
    MiniLM_RAG.py
  model_weights/
    transnetv2-pytorch-weights.pth # 29 MB - committed to git
  scripts/                         # offline ingestion (ASR x4, OCR x1, generic x4)
  utils/
    database_processing.py
    image_processing.py
    video_trans_detection.py
  .streamlit/
    config.toml                    # 1-line: maxUploadSize = 1000
  .gitignore                       # explicit allow-list for the TransNetV2 weight
```

Total: ~ 740 KB of source + 29 MB of weights, single commit, no tests visible, no CI config.

---

## 2. Architecture (theirs vs ours)

| Layer | Theirs (2025) | Ours (2026) | Notes |
|---|---|---|---|
| **UI** | Streamlit (147 KB monolith) | React + Vite + Zustand + WebSocket | [ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md). Their 1.5ù4 s/click was the documented root cause of last year's lower score. |
| **API tier** | FastAPI (107 KB) | FastAPI async + `asyncio.gather` | Keep the FastAPI split. Their tier was synchronous-heavy; we parallelise tools. |
| **Vector DB** | Qdrant (local + AWS cloud fallback) | Milvus 2.5 hybrid | Comparable; we chose Milvus per MEMORIA LSC'25 evidence. Qdrant cloud URL was hardcoded in `vector_store_optimized.py:90` ù third network dependency. |
| **Text index** | Elasticsearch with `standard` tokenizer + **`asciifolding`** filter | Elasticsearch + ICU + pyvi/underthesea | Their analyser strips diacritics. See ù3 and ù4.1. |
| **Default img enc** | `openai/clip-vit-base-patch32` (config.py:10, 512-dim, 2021) | SigLIP-2 So400m + Meta CLIP 2 ViT-H/14 (both 1024-dim) | Outdated baseline vs current SOTA. |
| **Other img enc** | SigLIP v1, BEiT-3, OpenCLIP | SigLIP **2** (Feb 2025), Meta CLIP **2** (Jul 2025) | Same family, ~2 generations newer. |
| **Reranker** | BLIP-2 (ITC/ITM), InternVL3, SigLIP2 | Vintern-3B-beta + Gemini Flash escalation | Their stack has no Vietnamese-native reranker. |
| **Planner / VLM** | GPT-4o + Phi-3 for query expand | SeaLLMs-v3-7B local + Groq escalation (under [bakeoff](../proposals/09-llm-path-bakeoff.md)) | Their hot path was cloud-only. |
| **OCR** | Gemini + GLM + Multi-LLM OCR (all cloud) | PaddleOCR PP-OCRv5 + VietOCR (local) | Three cloud dependencies vs zero. |
| **ASR** | Whisper API + Whisper VAD | PhoWhisper-large (WER 8.14 vs Whisper ~15) | Major Vietnamese ASR upgrade. |
| **Shot detect** | TransNetV2 PyTorch (weights in repo) | TransNetV2 (borrow, see ù3.4) | Direct reuse opportunity. |
| **Object detect** | YOLOE | YOLOv8 | Equivalent. |
| **Text embed** | BGE-M3 (via FlagEmbedding) | BGE-M3 | **Same choice** ù independent validation. |
| **Gen visual query** | Stable Diffusion (`StableDiffusion_genImg.py`) | SDXL (NII-UIT trick) | Same idea, we upgrade to SDXL. |
| **Quantization** | None | INT4 (AWQ) / FP4 ([ADR-0006](../adr/ADR-0006-int4-quantization-hot-path.md)) | Their inference target was unstated; ours is 12 GB. |
| **Auto-track agent** | None ù only `chatbot.py` single-shot LangChain | LangGraph state machine ([proposal 02](../proposals/02-automatic-track-agent.md)) | Greenfield for us. |

---

## 3. Patterns worth stealing

These five items are non-trivial wins, already de-risked by the previous team shipping them. Under [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md), we may borrow them under attribution.

### 3.1 DRES integration (HIGH VALUE ù saves ~2 engineer-days)

**Source**: `streamlit_api.py:21-200`
- Hardcoded DRES URL: **<https://eventretrieval.oj.io.vn>** ù this is the real production DRES server for AIC HCMC. Now we know.
- `LOGIN_API_URL`, `EVALUATION_LIST_URL`, `SUBMIT_API_URL` are at `/api/v2/login`, `/api/v2/client/evaluation/list`, `/api/v2/submit`.
- Session management lives in Streamlit session_state; we extract it into a clean Python client.
- Login returns `{sessionId, username, id, role}`.

**Where this lands in our repo**: [SPEC-0018](../specs/SPEC-0018-dres-integration.md). Implementation will be a clean `dres_client.py` module, not a Streamlit-coupled function set.

### 3.2 Pydantic request schemas (MEDIUM VALUE ù saves design iteration)

**Source**: `api/api_server.py:43-134`
- `KISSearchRequest`, `TextSearchRequest`, `ShotDetailRequest`, `SearchResponse`
- `EnhancedTemporalSearchRequest`, `TemporalCombinateSearchRequest`, `IntelligentSearchRequest`
- `EventSpec` per-event sub-schema with `kis / ocr / asr` text channels and `weights: {kis, ocr, asr}` for multimodal weighting

**Useful tunables they exposed** (these are well-thought-out and we should keep them):
| Knob | Default | Range | What it controls |
|---|---|---|---|
| `score_threshold` | 0.1-0.2 | [0, 1] | minimum cosine for a candidate to survive |
| `rrf_k` | 60 | [1, 1000] | RRF constant |
| `rrf_weights` | `None` | `{"siglip": ..., "beit3": ...}` | per-source RRF weighting |
| `rerank_top_k` | 100 | [5, 500] | how many to send to BLIP-2 |
| `rerank_score_type` | "itm" | "itc" / "itm" | speed vs precision toggle |
| `auto_expand` | True | bool | LLM paraphrase fan-out |
| `num_expansions` | 4 | [2, 10] | how many paraphrases |
| `per_event_k` | 30-50 | [10, 1000] | candidates per scene in TRAKE |
| `max_events` | 5 | [2, 8] | how many scenes |
| `decay_rate` | 0.01 | [0.001, 0.1] | ? for temporal scoring (their proto-DANTE) |
| `same_video_only` | True | bool | constrain TRAKE to single video |
| `max_gap_seconds` | None | optional | TRAKE temporal gap cap |

**Where this lands**: cross-references in SPEC-0008 (planner LLM), SPEC-0009 (tool registry), SPEC-0011 (DANTE DP for TRAKE) when those are authored.

### 3.3 BLIP-2 reranker ITC/ITM toggle (LOW VALUE but explicit prior art)

**Source**: `model/BLIP2_rerank.py` + `api_server.py:55`
The comment in the schema reads *"itc nhanh h?n; itm chùnh xùc h?n"* ("itc is faster; itm is more accurate"). They mapped the speed/quality trade-off. We can reuse the same toggle in [SPEC-0010](../specs/SPEC-0010-vlm-reranker.md) (VLM reranker), keeping it as a fallback when our default Vintern-3B-beta path is unavailable.

### 3.4 TransNetV2 wrapper + weights (DIRECT REUSE)

**Source**: `model/transnetv2_pytorch.py` (12.5 KB) + `model_weights/transnetv2-pytorch-weights.pth` (29 MB)
A PyTorch wrapper for TransNetV2 with the trained weights checked into git. The 29 MB file is small enough to vendor; the wrapper is a clean class interface. We borrow both into [SPEC-0003](../specs/SPEC-0003-data-ingestion.md) (data ingestion pipeline) when authored, with header attribution per [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md).

### 3.5 .gitignore weight-allowlist pattern (LOW VALUE ù adopt the policy)

**Source**: `.gitignore:90-103`
Their .gitignore uses an explicit allow-list for the one small TransNetV2 weight, blanket-ignoring everything else (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`). We adopt the same policy as a hard rule going forward to prevent the common "oops I committed a 4 GB checkpoint" failure.

### 3.6 Bookmarks UI primitive for TRAKE (CONCEPT REUSE)

**Source**: `streamlit_api.py:58-60`
```python
st.session_state.bookmarks = []  # [(video_name, frame_index), ...]
```
Their TRAKE staging used a session-state list of (video, frame) pairs ordered by tick time. The concept is correct ù operator-friendly for collecting 4 scenes ù and we rebuild it as a React drag-drop palette in [SPEC-0012](../specs/SPEC-0012-react-operator-console.md) when authored.

---

## 4. Independent validation of our Edge contributions

### 4.1 C1 DiacriticBERT ù the Vietnamese diacritic-stripping is documented, not hypothetical

`app/elasticsearch_service.py:32-43`:
```python
"multilingual_analyzer": {
    "type": "custom",
    "tokenizer": "standard",
    "filter": ["lowercase", "asciifolding", "stop"]
}
```

`asciifolding` literally removes Vietnamese diacritic marks during indexing. A query containing tones (`'/`, `''`, `'.`, `~`) loses its semantic precision the moment it hits this analyser. Their entire text retrieval over OCR and ASR transcripts ran on a diacritic-blind index. [ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md) C1 attacks this exact failure mode with a controlled-noise contrastive head. **The failure mode is not just plausible ù it shipped.**

### 4.2 C2 Per-task-type learned fusion ù they had globally-uniform RRF

`api_server.py:58-59`:
```python
rrf_k: int = Field(default=60, ge=1, le=1000)
rrf_weights: Optional[Dict[str, float]] = None
```

Optional per-query weight override, no per-task-type tuning, no LambdaRank, no learning. [ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md) makes C2 the default and RRF the fallback; the baseline shipping with the fallback as primary is independent evidence of the upside.

### 4.3 C4 Agent self-distillation ù they had no agent at all

`app/chatbot.py` is a single-shot LangChain over Qdrant top-3 with GPT-4o as the reader. It is not an agent. The automatic track is greenfield. [ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md) C4 has no competition.

---

## 5. Risks they accepted that we avoid

1. **Cloud OCR on hot path** (`Gemini_OCR.py`, `GLM_OCR.py`, `MultiLLM_OCR.py`) ù three providers, all cloud, all on the hot path. Venue network failure mid-round meant no OCR. Our PP-OCRv5 + VietOCR is local.

2. **GPT-4o as default LLM** (`config.py:11`) ù single-provider hot-path cloud dependency. Our SGLang + SeaLLMs-v3-7B is local; Groq is *escalation*, gated by [ADR-0005](../adr/ADR-0005-llm-path-bakeoff-gates-planner.md).

3. **Qdrant cloud fallback** hardcoded to an AWS URL (`vector_store_optimized.py:90`) ù third network dependency. We deploy Milvus on disk.

4. **No quantization** ù they had bigger GPUs, presumably. We have 12 GB ([ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md)) and INT4 ([ADR-0006](../adr/ADR-0006-int4-quantization-hot-path.md)) is mandatory.

5. **No Vietnamese tokenizer for the lexical layer** ù see ù4.1.

6. **No license discipline** ù `requirements.txt` mixes Gemini (proprietary), OpenAI (proprietary), GLM (model-specific terms), Qdrant cloud, GPT-4o, plus open-source licences. Our [strategy ù10 item 2](../strategy/00-master-strategy.md#10) tracks this explicitly.

7. **Single 147 KB Streamlit file** ù no modularity, no unit tests possible, no CI gate. [ADR-0009](../adr/ADR-0009-sdd-workflow.md) prevents this anti-pattern recurring.

---

## 6. Concrete actions

| # | Action | Where it lands | Effort |
|---|---|---|---|
| 1 | Port DRES login + submit flow | [SPEC-0018](../specs/SPEC-0018-dres-integration.md) | ~1 day |
| 2 | Vendor TransNetV2 wrapper + 29 MB weight | SPEC-0003 (when authored) | ~2 hours |
| 3 | Reuse Pydantic request schemas | SPEC-0008 / 0009 / 0011 (when authored) | ~2 hours per spec |
| 4 | Adopt their weight-allowlist `.gitignore` pattern | already in our `.gitignore` (commit `d6b3cb5`); cross-check | already done |
| 5 | Build C1 (DiacriticBERT) with extra confidence | SPEC-0014 (when authored) | unchanged ù but stronger motivation |
| 6 | 30-minute interview with the baseline author (`ThanhToan2111` ù current team member). Agenda + permission signoff bundled in [`docs/permissions/2025-baseline-reuse.md`](../permissions/2025-baseline-reuse.md) ù4. | team-lead task | 30 min call |

---

## 7. Open questions

> **Status update (2026-05-26):** All five questions below are **in flight**. The author of the 2025 baseline (`ThanhToan2111`) is a current team member; the 30-minute interview agenda at [`docs/permissions/2025-baseline-reuse.md`](../permissions/2025-baseline-reuse.md) ß4 maps each question to a discussion item. Update this section once the interview notes are filed.

- **Q1**: Is "ThanhToan2111" willing to share the actual 2025 dataset, ingestion outputs, or evaluation logs? A 30-minute conversation could save us weeks.
- **Q2**: Did the previous team's stack ever achieve sub-second per-interaction latency? If not, what was their measured p50/p95? Useful baseline for [SPEC-0001](../specs/SPEC-0001-evaluation-harness.md) CI thresholds.
- **Q3**: What was their actual finals score breakdown by task type? If TRAKE was where they bled the most, that biases [SPEC-0011](../specs/SPEC-0011-dante-dp-trake.md) priority upward.
- **Q4**: Is the production DRES URL (<https://eventretrieval.oj.io.vn>) public, or do they have credentials we can borrow for practice runs?
- **Q5**: Their `EnhancedTemporalSearchRequest.decay_rate: 0.01` and `max_gap_seconds: None` ù what values did they tune to in practice? Anchor for our DANTE ? sweep.

---

## 8. References

- Baseline repo: <https://github.com/ThanhToan2111/AIC_2026.git> (commit `c3c3545`, 2026-05-26)
- Borrowing policy: [`docs/adr/ADR-0010-borrow-from-2025-baseline.md`](../adr/ADR-0010-borrow-from-2025-baseline.md)
- Validated contributions: [`docs/adr/ADR-0007-original-contributions-c1-c2-c4.md`](../adr/ADR-0007-original-contributions-c1-c2-c4.md), [`docs/adr/ADR-0008-rrf-as-runtime-fallback.md`](../adr/ADR-0008-rrf-as-runtime-fallback.md)
- Hardware constraints: [`docs/adr/ADR-0003-rtx5070-finals-gh200-offline.md`](../adr/ADR-0003-rtx5070-finals-gh200-offline.md), [`docs/adr/ADR-0006-int4-quantization-hot-path.md`](../adr/ADR-0006-int4-quantization-hot-path.md)
- UI direction (against Streamlit): [`docs/adr/ADR-0004-no-streamlit-react-websocket-ui.md`](../adr/ADR-0004-no-streamlit-react-websocket-ui.md)
- LLM path: [`docs/adr/ADR-0005-llm-path-bakeoff-gates-planner.md`](../adr/ADR-0005-llm-path-bakeoff-gates-planner.md), [`docs/proposals/09-llm-path-bakeoff.md`](../proposals/09-llm-path-bakeoff.md)
- Workflow: [`docs/adr/ADR-0009-sdd-workflow.md`](../adr/ADR-0009-sdd-workflow.md), [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
