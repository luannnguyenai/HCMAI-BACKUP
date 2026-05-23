# Research Note 04 — Vietnamese AI Stack & Autonomous-Retrieval Agents (2026)

> Compiled May 2026. Two-part report: (1) Vietnamese-language NLP/VLM stack, (2) Agentic retrieval research for AIC2026's NEW automatic track.

---

# PART 1 — Vietnamese AI stack for multimedia retrieval

## 1.1 Vietnamese Vision-Language Models (VLMs)

### Vintern family (5CD-AI) — TOP VIETNAMESE VLM ?
- **Vintern-3B-beta**: #2 on MTVQA Vietnamese, beats GPT-4o on Vi tasks. <https://huggingface.co/5CD-AI/Vintern-3B-beta>
- **Vintern-3B-R-beta**: reasoning edition.
- **Vintern-1B-v2 / v3.5**: small fast Vi-only.
- **ColVintern-1B-v1**: ColBERT-style late-interaction Vietnamese document retriever — **UNIQUE: only Vietnamese ColPali equivalent**. Multi-vector embeddings of full pages ? MaxSim scoring. Perfect for slide/document-heavy frames.
- Architecture: based on InternVL2 architecture, fine-tuned on Vi data.
- HF org: <https://huggingface.co/5CD-AI>

### Qwen2.5-VL / Qwen3-VL — strong on Vietnamese, Apache-2.0
- Multilingual pretraining includes Vietnamese; Apache-2.0 for most sizes; native video.
- 7B fits 4090 BF16; 72B INT4 fits A6000.

### Gemma-2-9b-vi / Llama-3-Vietnamese / Qwen2.5-Vietnamese
- Community Vi fine-tunes (typically by VinAI or community teams). Quality varies.

## 1.2 Vietnamese LLMs (text only)

| Model | Params | Vietnamese benchmark | License |
|---|---:|---|---|
| **PhoBERT-large** (VinAI) | 370M | Vi RoBERTa baseline | MIT |
| **PhoBERT-v2** | 370M | better Vi benchmarks | MIT |
| **PhoGPT-4B / 7.5B / Chat** | 4B/7.5B | first Vi GPT | non-comm |
| **Vistral-7B-Chat** (VinAI/Viettel) | 7B | VMLU 50.07 | Llama 2 CL |
| **SeaLLMs-v3-7B-Chat** | 7B | **Vi M3Exam 64.9** (top open) | Apache-2.0 |
| **Sailor / Sailor 2** (SAIL) | 0.5B–14B | strong SEA langs | Apache-2.0 |
| **ViT5** | 220M–1B | T5-style for Vi | MIT |
| **BARTpho** | 396M | seq2seq | MIT |

### Recommendation
- **For function calling / agent driver**: SeaLLMs-v3-7B-Chat or Vistral-7B (function-calling tuned).
- **For Vietnamese ASR/OCR cleanup**: SeaLLMs-v3 is strong + cheap.

## 1.3 Vietnamese OCR

- **VietOCR** (pbcquoc): SOTA Vi-specific recognizer (10M-image internal: 0.880 full-seq precision). Apache-2.0. <https://github.com/pbcquoc/vietocr>
- **PaddleOCR Vietnamese** (latin recognizer in PP-OCRv5): 111-language coverage incl. Vi diacritics. Apache-2.0.
- **PaddleOCR-VL** (1.3B): 94.5% OmniDocBench v1.5; full doc understanding.
- **Vintern OCR mode**: VLM-based OCR for Vi tables/formulas.
- **Workflow**: PaddleOCR detection + VietOCR recognition via <https://github.com/kaylode/vnm-ocr-toolbox>.

## 1.4 Vietnamese ASR

- **PhoWhisper-large** (VinAI): WER **8.14** CMV-Vi, **4.67** VIVOS, **13.75** VLSP T1. **License: CC-BY-NC-SA-4.0** — verify with organisers!
- **PhoWhisper-medium / small / base / tiny**: progressive quality/speed tradeoff.
- **wav2vec2-large-vi-vlsp2020**: 317M, WER 8.61 VIVOS — beaten by PhoWhisper-medium.
- **Whisper large-v3-turbo** (OpenAI MIT): MIT-licensed fallback for English code-switched audio.

## 1.5 Vietnamese-English MT

- **VinAI envit5-translation**: BLEU 45.47 En?Vi, 40.57 Vi?En. <https://huggingface.co/VietAI/envit5-translation>
- **VinAI Translate**: paid API.
- **NLLB-200** (Meta): 200 lang, decent Vi but worse than envit5 on news.
- **MADLAD-400** (Google): 400 lang, strong on Vi.
- **Use case**: when you must translate Vi query to English for an English-only CLIP variant.

## 1.6 Vietnamese-aware CLIP

- **M-CLIP / OpenCLIP xlm-roberta-large-ViT-H-14** (LAION 5B): multilingual; baseline Vi-capable encoder.
- **SigLIP 2** (multilingual variants): strong on Vi via XM3600.
- **Meta CLIP 2 ViT-H/14**: **best XM3600 multilingual** in 2025 (Vi included). MIT.
- **ViCLIP-OT** (VietAI): KTVIC R@K 82.68%. Vi-specific.
- **dangvantuan/vietnamese-embedding**: used by MERVIN for transcript retrieval.

## 1.7 Vietnamese text retrievers / rerankers

- **BGE-M3** (multilingual; covers Vi strongly).
- **multilingual-e5-large-instruct**: MIRACL nDCG@10 66.5.
- **bkai-foundation-models/vietnamese-bi-encoder**: Vi-only SimCSE.
- **jina-embeddings-v3** (Vi-capable, but CC-BY-NC).
- **GTE-multilingual** (Apache-2.0).
- **PhoBERT-large fine-tuned**: still competitive for Vi-only.

## 1.8 Vietnamese tokenization

- **pyvi**: fast pure-Python. <https://github.com/trungtv/pyvi>
- **underthesea**: more accurate, more deps.
- **VnCoreNLP**: most accurate, Java backend.

## 1.9 Vietnamese datasets

- **OpenViVQA**: Vi visual QA.
- **ViTextVQA**: Vi scene-text VQA.
- **ViOCRVQA**: Vi OCR VQA.
- **ViCLEVR**: Vi compositional reasoning.
- **UIT-OpenViIC**: Vi image captioning.
- **KTVIC**: Vi image-caption benchmark.
- **VLSP 2023 ViLLM-Eval**: Vi LLM eval.
- **VLSP datasets** generally: Vietnamese Language and Speech Processing community.

---

# PART 2 — Autonomous / Agentic Retrieval

The AIC2026 has a NEW automatic-agent track. The closest prior art:

## 2.1 LSC '22-'26 trajectory
- LSC has run **automatic / autonomous sub-tracks** in 2024-25 as side experiments. Most "automatic" entries are simpler than interactive (no relevance feedback) and score significantly lower.
- Closest in-format reference: ACMMM 2025 News Event Retrieval — Automatic Mode (Vietnamese news data!) at <https://acmmm2025.org/grand-challenge/>.

## 2.2 SnapMind (MMM 2026) — recommended blueprint
- **Architecture**: LLM Planner takes (query, registry-of-components). Components = text, image, OCR, ADL, object retrieval modules. Planner outputs candidate execution plans (DAGs of component calls + fusion + normalization).
- **3 autonomy modes**: fully-autonomous / suggest-and-confirm / manual-with-suggestions.
- **Inspired by**: HuggingGPT-style tool routing, VideoAgent and VideoAgent2 for long-form video.
- Paper: <https://doi.org/10.1007/978-981-95-6963-2_20>

## 2.3 MARS @ CASTLE/EgoVis 2026
- GPT-5.4 decision agent iteratively chooses: (a) continue reasoning, (b) request a missing modality (gaze, heart-rate, photos, thermal), (c) answer, (d) random fallback.
- 0.57 accuracy on 185 Qs × 15 perspectives × 4 days.
- **Pattern**: agent as **evidence-router** over heterogeneous modalities.
- Paper: <https://arxiv.org/html/2605.18176v1>

## 2.4 Cascaded retrieve-then-rerank agent (arXiv 2512.12935)
- BEiT-3 + SigLIP for broad recall in Qdrant ? BLIP-2 cross-encoder for precise rerank.
- GPT-4o decomposes query into visual/OCR/ASR sub-queries with adaptive score fusion.
- Includes temporal-aware exponential-decay penalty for event-sequence coherence.
- Paper: <https://arxiv.org/pdf/2512.12935>

## 2.5 Smart routing for multimodal retrieval
- Learns *when* to search *what* modality.
- Paper: <https://arxiv.org/abs/2507.13374>

## 2.6 VideoSeek (CVPR 2026)
- Long-horizon video agent with 3 tool primitives: **scan / glance / zoom** — uses ~1/300 of frames vs full parsing.
- Repo: <https://github.com/jylins/videoseek>

## 2.7 Search-o1 (EMNLP 2025) — RAG with reasoning
- Reasoning model orchestrates retrieval steps; iterative refinement until confidence threshold.

## 2.8 MM-ReAct, VSearcher, OpenSearch-VL, HyperEyes, LMM-Searcher
- Different flavors of "ReAct over vision tools." All target VQA but transfer to retrieval.

## 2.9 VLM-as-judge reranking research
- **RagVL**: VLM scores candidate frames during RAG.
- **CoTRR**: chain-of-thought reranker; +10pt NDCG over single-shot.
- **UniRank**: unified reranker across modalities.
- **CIR-LVLM**: composed image retrieval with LVLM.

## 2.10 Test-time scaling for retrieval
- **SeerSC**: -47% tokens, -43% latency vs naive multi-step.
- **RPC** (Retrieval-Planning-Critique): three-stage loop.
- **TTR** (Test-Time Retrieval): generate K candidate queries, vote.

## 2.11 Reflexion + ReAct + RAR
- Established loop: act ? observe ? critique ? retry.

## 2.12 Agent framework comparison

| Framework | Strengths | Weaknesses | License |
|---|---|---|---|
| **LangGraph** | State machines, durable, mature | Verbose | MIT |
| **DSPy** | Programmatic prompts, optimizer | Niche | MIT |
| **smolagents** (HF) | Lightweight, code-agent | Less mature | Apache-2.0 |
| **PydanticAI** | Type safety, easy | Newer | MIT |
| **OpenAI Agents SDK** | Best for OpenAI ecosystem | OpenAI-centric | MIT |
| **LlamaIndex Workflows** | Tied to RAG ecosystem | Heavy | MIT |
| **Haystack** | Mature pipelines | Verbose | Apache-2.0 |
| **AutoGen / Magentic-One** | Multi-agent, conversational | Heavy | MIT |

**Recommendation for AIC2026**: **LangGraph** for the orchestration layer + **DSPy** for prompt optimization on the planner LLM. Familiar with Python team and integrates cleanly with FastAPI + Milvus + Vintern.

## 2.13 Recommended autonomous architecture for AIC2026

```
User query (text or multimodal hint)
        ?
        ?
????????????????????????????????????
? Planner LLM                      ? ? SeaLLMs-v3-7B or Vistral-7B-FC
?  (function-calling)              ?   (or Gemini 2.5 Flash for budget)
?  - parse query ? JSON intent     ?
?  - decide which tools to invoke  ?
?  - emit execution plan (DAG)     ?
????????????????????????????????????
              ?
              ?
????????????????????????????????????
? Tool registry (parallel exec)    ?
?  ? text_retrieval(SigLIP-2)      ? — primary visual recall
?  ? vi_text_retrieval(MetaCLIP2)  ? — Vietnamese-optimized
?  ? ocr_search(BGE-M3 + BM25)     ? — proper nouns
?  ? asr_search(BGE-M3 + BM25)     ? — speech content
?  ? object_filter(YOLO/Qwen-VL)   ? — structured filter
?  ? scene_filter(Places365)       ? — structured filter
?  ? temporal_filter(time window)  ? — structured filter
?  ? image_query(SigLIP-2 image)   ? — image-by-image
?  ? generative_visual_query        ? — Stable Diffusion ? search
?                                  ?   (NII-UIT VBS'25 trick)
????????????????????????????????????
              ?
              ?
????????????????????????????????????
? Score normalisation + RRF fusion ?
????????????????????????????????????
              ?
              ?
????????????????????????????????????
? Temporal coherence (TRAKE)       ? — DANTE DP for 4-scene
?  - SeqWin (tight) / ParChain     ?
?  - exponential-decay penalty ?   ?
????????????????????????????????????
              ?
              ? top-30
????????????????????????????????????
? VLM-as-judge rerank              ? ? Vintern-3B-beta (Vi)
?  - 3×3 grid prompt + CoT         ?   or Gemini 2.5 Flash
?  - emit final rank + confidence  ?
????????????????????????????????????
              ?
              ?
????????????????????????????????????
? Decision agent                   ?
?  - confident ? submit            ?
?  - low conf ? critique + retry   ?
?  - max-iter ? fallback heuristic ?
????????????????????????????????????
```

Key design choices vs interactive system:
1. **No human-in-the-loop** ? the LLM planner must internalize all the heuristics the human operator would use.
2. **Confidence calibration is critical** ? train/select a model with reliable score distributions; use ensemble of K planner runs with self-consistency.
3. **Reduce token cost** ? cache LLM plans per query type; use cheap model (Gemini 2.5 Flash, SeaLLMs-v3) for routine queries, escalate to GPT-4o/Gemini 3 Pro for hard ones.
4. **Self-verification** ? after submitting a candidate, the VLM-judge re-examines and may retract / resubmit.
5. **Time budget** ? the automatic track will have a hard wall-clock cap per query (likely 30-60s). Plan parallel execution from the start.

---

# Sources

- HCMC AIC official: <https://aichallenge.hochiminhcity.gov.vn/>
- 5CD-AI Vintern: <https://huggingface.co/5CD-AI>
- VinAI VietAI envit5: <https://huggingface.co/VietAI/envit5-translation>
- PhoWhisper: <https://github.com/VinAIResearch/PhoWhisper>
- VietOCR: <https://github.com/pbcquoc/vietocr>
- BGE-M3: <https://huggingface.co/BAAI/bge-m3>
- Vietnamese embedding (MERVIN): <https://huggingface.co/dangvantuan/vietnamese-embedding>
- SeaLLMs-v3: <https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat>
- Vistral-7B: <https://huggingface.co/Viet-Mistral/Vistral-7B-Chat>
- SnapMind: <https://doi.org/10.1007/978-981-95-6963-2_20>
- MARS @ CASTLE: <https://arxiv.org/html/2605.18176v1>
- Cascaded multimodal agent: <https://arxiv.org/pdf/2512.12935>
- Smart routing: <https://arxiv.org/abs/2507.13374>
- VideoSeek: <https://github.com/jylins/videoseek>
- ACMMM 2025 News Event Auto Mode: <https://acmmm2025.org/grand-challenge/>
- LangGraph: <https://github.com/langchain-ai/langgraph>
- DSPy: <https://github.com/stanfordnlp/dspy>
- smolagents: <https://github.com/huggingface/smolagents>
- HCMC AIC 2025 EEIoT_newbie: <https://arxiv.org/html/2512.06334>
- MERVIN: <https://arxiv.org/html/2605.16120v1>
- QUEST + DANTE: <https://arxiv.org/html/2512.13169>
