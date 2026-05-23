# Research Note 01 — AI Challenge HCMC: Prior Editions (2023-2025)

> Compiled from web research on May 24, 2026. Use this to inform AIC2026 architecture choices that match what has actually worked at this specific competition. Every load-bearing claim is cited.

---

## 0. Organizer & history

The **H?i thi Th? thách Trí tu? Nhân t?o TPHCM** (HCMC AI Challenge) is organised annually by **S? Khoa h?c và Công ngh? TPHCM** with **?HQG-HCM** (University of Science, VNU-HCM) and HCA. It has been run since 2020. **Since 2022**, the competition has used the **LSC/VBS interactive-retrieval format**: teams build retrieval systems and run them live against a held-out task set, scored on a server.

- Two divisions every year: **B?ng A** (university / open) and **B?ng B** (high school).
- Past finals venues: Saigon Innovation Hub, HCMUS, etc. Live scoring with mixed elimination (offline submission) + finals (interactive on DRES-like server).
- Special session at SoICT (Vietnamese symposium on Information & Communication Technology) publishes the top team papers in ACM proceedings.

## 1. AIC HCMC 2023

- **Problem**: Event retrieval from **Vietnamese news broadcast videos**. KIS-style ("known-item search") plus partial ad-hoc. Vietnamese text + Vietnamese ASR transcripts critical.
- **Dataset**: ~300 hours of Vietnamese news / public TV; pre-extracted keyframes (~1.5M); transcripts and OCR provided in starter kit.
- **Evaluation server**: CodaLab for the elimination round; DRES-like custom server for finals (live submission, scored).
- **Scoring**: Top-k Recall Score formula (preview of the 2025 Mean-of-Top-k R-Score).
- **B?ng A winner**: **ToMS retrieval** (UIT/HCMUS-affiliated team), CLIP ViT-B/32 + **Temporal-Ordered Multi-Query Scoring (ToMS)** + transcript BM25. Public repo: <https://github.com/ziap/toms-retrieval>.
- **B?ng B winner**: high-school team with a CLIP-based GUI; code less public.

### Architecture pattern that worked in 2023
1. **TransNetV2** shot detection ? 1 keyframe per shot (~1.5M frames total).
2. CLIP ViT-B/32 image embeddings ? FAISS IndexFlatIP.
3. ASR transcript + OCR indexed in Elasticsearch with **Vietnamese tokenizer** (pyvi).
4. Temporal-ordered multi-query: for 2+ scene queries, find optimal sequence of frames respecting time order.
5. Browser-side GUI with grid + sort modes + temporal context expansion.

## 2. AIC HCMC 2024

- **Problem**: Extended KIS + **Q&A track** (new) + ad-hoc. Vietnamese news/TV again. Multi-modal (image + Vietnamese audio transcript + Vietnamese OCR).
- **Dataset**: Bigger than 2023; estimated ~500 hours of broadcast video.
- **B?ng A winner**: **TycheVid** (UIT) — extended pipeline with **OpenCLIP ViT-H/14** + BLIP-2 + Vietnamese-aware reranking.
- **B?ng B winner**: **FriedPotatoes** (PTNK high school) — slimmed UI, CLIP ViT-L baseline.

### Notable systems (SoICT 2024 proceedings, [Springer 10.1007/978-981-96-4291-5](https://link.springer.com/book/10.1007/978-981-96-4291-5))
- **KPI (UIT)** — Knowledge-based retrieval — combined text + ASR + OCR + temporal filters + dominant-color search + user-feedback UI. ([Chapter 7](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_7))
- **Vo et al.** — CLIP ViT-L + **TASK-Former** (hybrid sketch+text query) + transcript + OCR. ([Chapter 19](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_19))

### Known pitfalls in 2024
- Teams that did **not** prepare a Q&A pipeline (RAG / VLM reader) lost ~30% of total points on the new track.
- Pure CLIP-only systems missed Vietnamese proper-noun queries — Vietnamese OCR/ASR text retrieval was decisive.

### Public repos
- <https://github.com/trnKhanh/AIC24>
- <https://github.com/duongngockhanh/event-retrieval-from-video>
- <https://github.com/AIVIETNAMResearch/VN_Multi_User_Video_Search>

## 3. AIC HCMC 2025

- **Problem**: KIS + Q&A + **TRAKE** (new — Temporal **R**etrieval of **A**ligned **K**ey-frame **E**vents). TRAKE asks teams to return a sequence of 4 keyframes that match a 4-scene query in correct temporal order.
- **Dataset**: ~800 hours of Vietnamese broadcast; keyframes pre-extracted; ASR + OCR provided.
- **Scoring (official 2025 formula)**: Mean of Top-k R-Scores over k ? {1, 5, 20, 50, 100}. This rewards both precision-at-1 and recall.
- **B?ng A winner**: **OpenCubee_1** (UIT) — **ConvAgent** system (LLM-based conversational query planner over the retrieval components). Code not yet released as of May 2026.
- **B?ng B winner**: **WuDButterflies** (THPT Ngô Quy?n) — well-engineered CLIP+OCR+ASR pipeline with a clean UI.

### Notable systems
- **AIO_Owlgorithms — QUEST + DANTE** ([arXiv 2512.13169](https://arxiv.org/html/2512.13169)): DANTE-style dynamic-programming for TRAKE — minimum-cost path through 4-scene candidates with a temporal-distance penalty ? ? [0.001, 0.01].
- **MERVIN** ([arXiv 2605.16120](https://arxiv.org/html/2605.16120v1)) — used **Meta PE-Core-bigG-14-448** (image), Milvus, **Gemini-cleaned Whisper ASR**, `dangvantuan/vietnamese-embedding` for transcript. Final score 79/88 on B?ng A.
- **MemoriEase 3.0** influence (DCU/UIT collaboration).

### Architecture trend in 2025
1. **TransNetV2 shot detection** ? keyframes (~1M to ~3M scale).
2. Image embeddings: **PE-Core-bigG-14-448** or **BEiT-3** dominant (replaces CLIP ViT-H/14 of 2024).
3. Vector DB: **Milvus** with hybrid (dense + structured filter) replaced FAISS in most top systems.
4. ASR: **Whisper-large-v3** transcripts then **Gemini 1.5 Flash** cleanup (fix grammar, restore diacritics, segment sentences).
5. OCR: PaddleOCR Vietnamese + VietOCR fallback.
6. Vietnamese text retriever: `dangvantuan/vietnamese-embedding` or `bkai-foundation-models/vietnamese-bi-encoder`, indexed in **Elasticsearch** with pyvi/underthesea tokenizer.
7. TRAKE: **DANTE dynamic-programming** over 4-scene candidate lists with temporal-distance penalty.
8. **LLM query rewriting** (Gemini / GPT-4o) for OOK (out-of-knowledge) named entities CLIP can't embed.
9. UI: React with submission/verification page, keyframe scrubber, temporal-link arrows.

### Common pitfalls in 2025 (that cost points)
- **OOK entities**: Queries mentioning specific Vietnamese celebrities, brands, places that frozen CLIP doesn't know. Mitigation: LLM query rewrite + external image search to get a visual prior.
- **TRAKE collapse**: Pure semantic search scatters the correct 4-scene sequence across non-adjacent timestamps. Mitigation: DANTE-style DP with tuned ?.
- **ASR transcription errors**: Whisper Vietnamese ASR has many diacritic errors; teams that didn't clean transcripts (Gemini Flash post-processing) lost matches on proper-noun queries.

## 4. Synthesis & 2026 Implications

### 4.1 The de-facto evaluation server
- Elimination round: **CodaLab** (offline, submit JSON of (video_id, frame_id) tuples).
- Finals: **DRES-style live server** (custom HCMUS-hosted; submit through a web UI on a clock).

### 4.2 The battle-tested 9-step architecture template
| Step | 2025 best-of-breed | Notes |
|---|---|---|
| 1. Shot/scene segmentation | TransNetV2 + KDE-GMM frame sampling | Dense ~1 keyframe per 2-3s |
| 2. Image embedding | PE-Core-bigG-14-448 OR BEiT-3 OR OpenCLIP ViT-H | Vietnamese-news-tuned |
| 3. Vector index | Milvus hybrid (dense + filter) | Replaced FAISS |
| 4. ASR | Whisper-large-v3 + Gemini Flash cleanup | OR PhoWhisper-large |
| 5. OCR | PaddleOCR Vietnamese + VietOCR | Critical for proper nouns |
| 6. Text retrieval | Vietnamese bi-encoder + BM25 (pyvi) in Elasticsearch | Hybrid ?-fusion |
| 7. Temporal/TRAKE | DANTE DP with ? ? [0.001, 0.01] | New for 2025; carried into 2026 |
| 8. LLM query rewrite | Gemini 1.5/2.5 Flash or GPT-4o | OOK handling |
| 9. UI | React grid + temporal-link UI + verification page | Operator speed matters |

### 4.3 Gap analysis for the 2026 automatic track (NEW for AIC2026)
- **Automatic track** has *no precedent* in HCMC AIC editions 2023–2025. Closest references:
  - **ACMMM 2025 News Event Retrieval Grand Challenge** Automatic Mode (Vietnamese news source) — closest twin.
  - **SnapMind** (MMM 2026, [doi.org/10.1007/978-981-95-6963-2_20](https://doi.org/10.1007/978-981-95-6963-2_20)) — LLM Planner over retrieval components.
  - **MARS @ CASTLE/EgoVis 2026** ([arXiv 2605.18176](https://arxiv.org/html/2605.18176v1)) — VLM evidence-router across modalities.
- **Implication**: The team that builds a clean, modular tool-router agent on top of a strong interactive retrieval substrate will dominate the automatic track because most other teams will tack on an LLM as an afterthought.

### 4.4 Known unknowns (still need verification)
- **Exact 2026 dataset**: not yet released (preliminary round details due 25 June 2026 per official site).
- **Exact 2026 scoring**: organisers may keep 2025's formula or add a separate automatic-track score.
- **Will TRAKE remain?** Likely yes — it's the natural extension toward multi-step temporal reasoning.
- **How many tasks per round?** Historically: ~24 finals tasks (8 KIS + 8 QA + 8 TRAKE/Ad-hoc) over a 3-hour session.

## 5. Sources
- AIC HCMC official: <https://aichallenge.hochiminhcity.gov.vn/>
- SoICT 2024 proceedings: <https://link.springer.com/book/10.1007/978-981-96-4291-5>
- LSC 2022-24 SOTA review (foundational): <https://arxiv.org/abs/2506.06743>
- ToMS Retrieval (2023 winner): <https://github.com/ziap/toms-retrieval>
- MERVIN (2025): <https://arxiv.org/html/2605.16120v1>
- QUEST + DANTE: <https://arxiv.org/html/2512.13169>
- KPI (2024): <https://link.springer.com/chapter/10.1007/978-981-96-4291-5_7>
- AIVietnam multi-user: <https://github.com/AIVIETNAMResearch/VN_Multi_User_Video_Search>
- AIC24 sample: <https://github.com/trnKhanh/AIC24>
