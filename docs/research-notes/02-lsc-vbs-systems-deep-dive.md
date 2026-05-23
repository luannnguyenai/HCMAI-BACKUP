# Research Note 02 — LSC & VBS Top Systems Deep Dive (2022-2025)

> Synthesis of the world's two best-known interactive multimedia retrieval competitions. AIC2026 inherits the LSC/VBS task design directly. Use this as the master technical reference. Every system is cited with paper + code URLs where available.

---

## 1. Final standings (top-3) — what to study first

### LSC

| Year | 1st | 2nd | 3rd |
|---|---|---|---|
| LSC'22 (Newark) | MyScéal | LIFEXPLORE | LifeSeeker 4.0 |
| LSC'23 (Thessaloniki) | LIFEXPLORE | MyEachtra | Memento |
| LSC'24 (Phuket) | LIFEXPLORE | SnapSeek | VISIONE |
| LSC'25 (Chicago) | **MEMORIA** | **SnapSeek 3.0** | **MemoriEase 3.0** |

### VBS

| Year | 1st | 2nd | 3rd |
|---|---|---|---|
| VBS'22 (Online) | Vibro | CVHunter | VISIONE |
| VBS'23 (Bergen) | Vibro | VISIONE | CVHunter |
| VBS'24 (Amsterdam) | VISIONE 5.0 | Vibro | PraK Tool |
| VBS'25 (Nara) | **NII-UIT (Vietnamese!)** | **PraK Tool V3** | **diveXplore** |

Notable: **NII-UIT** (UIT-Vietnam + NII-Japan) won VBS'25. Vietnamese teams already operate at the absolute frontier of this format.

## 2. Datasets & scoring

### LSC dataset
- 725K first-person Narrative Clip images (1024×768, faces blurred), 18 months Jan-2019–Jun-2020.
- Per-minute biometrics (heart rate, steps, sleep), GPS, COCO objects, Places365 scenes, OCR, VAISL semantic locations, flight data.

### Tasks
- **KIS** (5 min, 30s incremental hints, single image): `score_KIS = max(0, 100 ? 50·t/T ? 10·w)`
- **Ad-hoc** (3 min, unlimited submissions): `score_Ad-hoc = 100 · correct/(correct + incorrect/2) · correct/total`
- **QA** (3 min, single text answer, human-judged)

Live evaluation by **DRES** server: [doi.org/10.1145/3678881](https://doi.org/10.1145/3678881).

## 3. LSC system deep-dives

### 3.1 SnapSeek / SnapSeek 2.0 / SnapSeek 3.0 (HCMUS / Vietnam — 2nd LSC'25)
- **Embedding ensemble**: OpenCLIP ViT-H/14 + BLIP-2 + BEiT-3 (v1), added **CLIPS** + Qwen2.5-VL scene-graph triplets in v3.
- **Index**: Milvus (sparse + dense fields). Hybrid query, client-side Fuse.js fuzzy.
- **Temporal**: LLM parses query ? {before, after, interval{min, max}}. **SeqWin** (forward window, score = mean of pair) and **ParChain** (independent chains).
- **Side channels**: OCR + Elasticsearch fuzzy, Places365 location, **LSC-ADL 35 ADL classes** ([arXiv 2504.02060](https://arxiv.org/abs/2504.02060)) via semi-automatic clustering.
- **LLM**: Qwen2.5-VL-7B local scene-graph + temporal-clause extraction.
- **UI**: ?80% viewport on images, 3 view modes (ranked list / grouped-by-day / contextual line view), whiteboard tool.
- **Result LSC'25**: 2711/3000 (KIS 1000, QA 781, Ad-hoc 930).
- **Paper**: [dl.acm.org/doi/10.1145/3729459.3748697](https://dl.acm.org/doi/10.1145/3729459.3748697)

### 3.2 MEMORIA (Aveiro — LSC'25 winner)
- **Pipeline**: Annotation (objects, scenes, OCR, captions) ? BLIP-2 vectors ? Milvus.
- **Winning change vs LSC'24**: swap free-text graph DB ? Milvus.
- **LLM**: query parser for keyword extraction + misspelling correction.
- **UI**: Vue/React filter-rich grid.
- **Lesson**: Authors note "BLIP feature was not as helpful as we initially thought" — the win was the vector-DB infrastructure.
- **Paper**: [dl.acm.org/doi/10.1145/3729459.3748693](https://dl.acm.org/doi/10.1145/3729459.3748693)

### 3.3 MemoriEase 3.0 (DCU — LSC'25 3rd, 2nd QA)
- **Two-phase**: offline embedding (BLIP-2 + CLIP ViT-L/14) ? Elasticsearch with kNN vector fields.
- **Fusion**: weighted average of two cosines, weights tuned on validation.
- **Critical decision**: deliberately **no event grouping** — events conflated semantically distinct frames.
- **QA**: GPT-o1 RAG with BERT cross-encoder reranker (fine-tuned on Q-description pairs). Heuristic: visual Q ? top-10 images; metadata Q ? top-30-50 with reranked descriptions.
- **Innovation**: pool-embedding = ?·query + (1-?)·mean(pooled image embeddings) for relevance feedback that preserves query semantics.
- **Paper**: [doras.dcu.ie/31771/](https://doras.dcu.ie/31771/1/3729459.3748689.pdf)

### 3.4 LIFEXPLORE (Klagenfurt — LSC'23 & LSC'24 winner)
- **Pipeline**: Day-folder ? 5fps transcoded videos ? optical-flow shot detection ? keyframe per shot ? OpenCLIP ViT-H/14 LAION-2B + legacy Inception-BN concepts.
- **Index**: FAISS (HNSW/IVF) + Milvus.
- **Filter containers**: weekday, time-of-day, location, all auto-combined.
- **UI signature**: hierarchical 2-D feature map of keyframes ordered by Self-Sorting-Map criteria.
- **No LLM** (deliberately).
- **Weakness identified**: verification overhead — experienced operator spent ~4 min on verification of an already-correct hit.
- **Paper**: [doi.org/10.1145/3643489.3661119](https://doi.org/10.1145/3643489.3661119)

### 3.5 MyScéal / MyEachtra / MyEachtraX (DCU — LSC'22 winner)
- **Pipeline**: Event-centric — content-based segmentation, per-event aggregated embeddings (OpenCLIP ViT-H/14), Elasticsearch.
- **LLM**: InternLM-XComposer2 local + ChatGPT for QA.
- **MyEachtraX QA accuracy**: 72.2% on official LSC'24 questions. Bottleneck = event retrieval recall, not the reader.
- **Paper**: [doras.dcu.ie/30727/](https://doras.dcu.ie/30727/1/3643489.3661128%20%281%29.pdf)

### 3.6 VISIONE 5.0 (CNR Pisa — VBS'24 winner, LSC'24 3rd)
- **Embeddings**: **OpenCLIP ViT-L/14 LAION-2B + CLIP2Video + ALADIN**, late fusion.
- **Unique trick**: **ALADIN encoded via Surrogate Text Representation (STR)** — dense vectors encoded as text tokens, indexed in **Apache Lucene**, retrieved via BM25. Most cost-efficient large-scale dense index in the community.
- **Side channels**: object/color **canvas** (rectangle of given color at given position).
- **For LSC**: used **only visual content**, no metadata.
- **Code**: <https://github.com/aimh-lab/visione>
- **Paper**: [zenodo.org/records/13903347](https://zenodo.org/records/13903347)

### 3.7 vitrivr / vitrivr-engine / vitrivr-VR (Basel + UZH + DCU)
- **Pipeline**: Cottontail/PostgreSQL + Cineast/vitrivr-engine + vitrivr-ng or VR frontend.
- **Embeddings (LSC'24)**: **OpenCLIP ViT-B/32 with XLM-RoBERTa multilingual text encoder** (`CLIP-ViT-B-32-xlm-roberta-base-laion5B`) — supports non-English queries.
- **vitrivr-engine**: modular REST/OpenAPI, JSON schema config. <https://github.com/vitrivr/vitrivr-engine>
- **VR variant**: spatiotemporal calendar/map; CollaXRSearch is a VR collaboration fork.

### 3.8 Memento 4.0 (DCU)
- **Ensemble**: weighted CLIP (ResNet50x64 + ViT-L/14).
- **LLM**: GPT-3.5 Turbo + Mistral 7B in RAG QA pipeline grounded on event-level summaries.
- **Paper**: [doras.dcu.ie/30691/](https://doras.dcu.ie/30691/)

### 3.9 Libro (visual-computing.de)
- **Embedding**: **EVA-CLIP ViT-E** joint embeddings + **Exploration Graph** for proximity search.
- **Trick**: binarized embeddings + Hamming distance for sub-ms retrieval.
- **OCR**: BM25 + hybrid text-image alpha-slider.
- **UI**: Dynamic grid + **FLAS** sorting for visual similarity ? demonstrated **fastest KIS time-to-first-correct** in LSC'24.
- **Paper**: [doi.org/10.1145/3643489.3661124](https://doi.org/10.1145/3643489.3661124)

### 3.10 LifeGraph 3/4/5 (UZH)
- **Backend**: MeGraS multimodal knowledge graph ([MM '25, 10.1145/3746027.3756872](https://doi.org/10.1145/3746027.3756872)) — RDF/SPARQL with k-NN extensions.
- **Embeddings**: OpenCLIP ViT-H/14 + BLIP-2/LLaVA captions/relations.
- **UI**: React ? visual SPARQL builder. <https://github.com/MediaGraphOrg/LifeGraph5>

### 3.11 Voxento-Pro (DCU)
- Voice ? Whisper ? embedding search ? OpenAI Assistant.
- Conversational hands-free; flight-data contributor.

### 3.12 PraK Tool V2/V3 (Charles Univ — VBS'25 2nd)
- **Architecture**: stateless data-service ? frontend; modular async pipelines.
- **V3 innovations**: grid-based localized text query (region of frame), texture-based queries, Bayesian relevance feedback, `/temporalQuery` endpoint.
- **Variance lesson**: PraK1/2/3 instances varied wildly in LSC'24 (93 vs 63 on AD05) — **strong operator-dependence**.
- **Thesis**: [dspace.cuni.cz/.../120518844.pdf](https://dspace.cuni.cz/bitstream/handle/20.500.11956/202882/120518844.pdf)

### 3.13 EAGLE & CollaXRSearch (UIT)
- **EAGLE**: eye-gaze implicit relevance feedback.
- **CollaXRSearch**: shared VR workspace.
- Both contribute clever UX rather than retrieval quality.

### 3.14 Exquisitor (Reykjavík)
- Stateful **multi-turn user relevance feedback** over CLIP. VBS'25 4th.

## 4. VBS system deep-dives

### 4.1 VISIONE 5.0 — VBS'24 winner
- Same engine as LSC version; dockerized microservices with GPU.
- **STR + Lucene** = key cost lever.
- Whisper ASR + object/color canvas + DINOv2 visual-similarity browse.

### 4.2 Vibro (HTW Berlin — VBS'22, '23 winner; Best-Textual-KIS expert 2024)
- **2024 embeddings**: CLIP ViT-L/14@336 + **EVA-CLIP ViT-E** for text-based + **MixedSwim** for image-based.
- **Model selection**: chosen offline by replaying KIS logs against candidates.
- **PCA-compressed float vectors in memory**.
- **FLAS sorting** = "best-arrangement quality and lowest latency" per VBS'24 report.
- **Paper**: [files.visual-computing.com/research/Vibro_Video_Browsing_with_Semantic_and_Visual_Image_Embeddings.pdf](https://files.visual-computing.com/research/Vibro_Video_Browsing_with_Semantic_and_Visual_Image_Embeddings.pdf)

### 4.3 NII-UIT — VBS'25 winner (Vietnamese-led team)
- **LLM strategies**:
  - (1) LLM **paraphrases** the natural-language query into multiple variants ? embedding matched against pre-stored keyframes.
  - (2) LLM converts free-text to an image via **Stable Diffusion** ? similarity-search ? **generative visual query**, a unique trick.
- **Temporal**: "Dynamic temporal search" — frame relevance is re-evaluated against neighboring frames instead of fixed-window.
- **Paper**: [Springer 10.1007/978-981-96-2074-6_38](https://doi.org/10.1007/978-981-96-2074-6_38)
- **Implication**: **Vietnamese teams already invented and deployed the generative-visual-query trick.** AIC2026 teams should expect this to be a baseline.

### 4.4 diveXplore (Klagenfurt — VBS'25 3rd)
- **Backend (IViSE 2025)**: TransNet V2 + EasyOCR/CRAFT + Whisper + OpenCLIP ViT-H/14 LAION-2B.
- **Middleware**: MongoDB + FAISS + Node.js + WebSocket.
- **Frontend**: Angular SPA.
- **UI signature**: **keyframe scrubbing** (shift+hover scrubs a low-GOP preview), temporal `q1 < q2` syntax.
- **Paper**: [CVPR 2025 IViSE](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Schoeffmann_AI-based_Video_Content_Understanding_for_Automatic_and_Interactive_Multimedia_Retrieval_CVPRW_2025_paper.pdf)

### 4.5 ViewsInsight 2.0 (UIT) and VideoEase (DCU)
- **ViewsInsight 2.0**: automatic LLM query generator ? multiple paraphrases. ([Springer 10.1007/978-981-96-2074-6_45](https://doi.org/10.1007/978-981-96-2074-6_45))
- **VideoEase (VBS'25)**: three-model ensemble (CLIP ViT-L/14@336 + BLIP-2 + OpenCLIP ViT-L/14 LAION-2B) in **Milvus hybrid index** (native multi-vector + filter). ([doras.dcu.ie/30858/](https://doras.dcu.ie/30858/1/MMM_VBS25_Linh.pdf))

### 4.6 VIREO (CityU HK)
- Free-text + CLIP4Clip + ITV + multimodal cross-encoder. LLM-assisted query reformulation for KISC (2025). [Springer 10.1007/978-981-96-2074-6_36](https://doi.org/10.1007/978-981-96-2074-6_36)

### 4.7 HORUS, Fusionista, VERGE, VEAGLE, MediaMix
- **HORUS**: pure-MLLM end-to-end retrieval, early example. [Springer 10.1007/978-981-96-2074-6_34](https://doi.org/10.1007/978-981-96-2074-6_34)
- **Fusionista**: depth-aware embeddings (3-D scene reconstruction fused with CLIP).
- **VEAGLE**: VBS port of EAGLE eye-gaze concept.
- **MediaMix**: multimedia retrieval in mixed reality (HoloLens).

## 5. Cross-cutting patterns (the 2025-2026 best-of-breed pipeline)

| Layer | Best-of-breed 2025-26 |
|---|---|
| Shot segmentation (video) | **TransNetV2** |
| Keyframe selection | middle of shot + uniform 1fps sampling |
| Primary embedding | **OpenCLIP ViT-H/14 LAION-2B** or **EVA-CLIP ViT-E** or **PE-Core-bigG-14-448** (best Vietnamese) |
| Secondary embedding | **BLIP-2** for caption/QA + **BEiT-3** for fine-grained |
| Indexing | **Milvus hybrid** (vector + filter in one query) |
| Fusion | late fusion / **RRF** of per-model rankings; **STR + Lucene** is the cheapest dense at scale |
| Temporal | LLM-parsed `before/after/interval` JSON ? SeqWin (tight) or ParChain (loose) |
| OCR / ASR | EasyOCR + CRAFT + Whisper-large-v3 (or **PhoWhisper** for Vietnamese) |
| Object / place / ADL | YOLO + Places365 + custom ADL (35 classes) |
| LLM | GPT-4o/o1 for QA RAG; Qwen2.5-VL-7B local for scene-graph & captions; Mistral-7B as cost fallback |
| Reranker | BLIP-2 ITM head OR fine-tuned BERT cross-encoder OR VLM-as-judge (Gemini 2.5 Flash) |
| UI essentials | activity-clustered timeline; group-by-day; keyframe scrubbing; `q1 < q2` temporal syntax; +/? relevance feedback; query history |
| Eval infra | self-hosted **DRES** |

### What separated #1 from middle-of-pack
1. **Ensemble + smart fusion** > single-model
2. **Filter-first reduction** (ADL labels, time bucket, location) before semantic search
3. **Operator skill** — PraK1/PraK2 differed by 30 points on the same engine
4. **UX speed** dominates KIS times — Libro's proximity-graph and diveXplore's scrubbing shave seconds ? score points under 50·t/T penalty
5. **MEMORIA's LSC'25 win** = single decisive infra swap (graph DB ? Milvus)

### Task-specific winning recipes
- **KIS**: Strong CLIP family + fast browse + temporal `<` chain. Speed > recall.
- **Ad-hoc**: Recall is strongest predictor (Spearman 0.75) ? high-recall ensembles + relevance feedback.
- **QA**: Embedding retrieval ? cross-encoder rerank ? **RAG with GPT-4o/o1** on event-level descriptions. Visual-only QA ? skip captions, hand images to MLLM.

### Pitfalls (LSC review §IV-D)
1. Single-model systems plateaued — Spearman embedding?score correlation went from 0.40-0.48 (2022) to non-significant (2023-24) as everyone adopted them.
2. **Event grouping is a trap** if not careful — MemoriEase reverted to per-image after frames got conflated.
3. **Pure-LLM conversational didn't dominate** — top QA systems were NOT explicitly conversational LLM. RAG wins; chat alone does not.
4. **Multi-instance same-system** showed huge variance (PraK1 vs PraK2 ±30 pts). Train operators on ONE system rather than diversify.
5. MEMORIA's BLIP-only embeddings showed mode collapse ("retrieved almost always same images") — ensemble or risk it.

## 6. The autonomous-agent track (new for AIC2026)

### 6.1 SnapMind (MMM 2026) — blueprint
- **LLM Planner** takes (query, registry-of-components) ? outputs candidate execution plans (DAGs of component calls + fusion + normalization).
- Three autonomy modes: fully autonomous / suggest-and-confirm / manual-with-suggestions.
- Score normalization + rank fusion. Users can edit/discard plans mid-flight.
- **Paper**: [doi.org/10.1007/978-981-95-6963-2_20](https://doi.org/10.1007/978-981-95-6963-2_20)

### 6.2 ACMMM 2025 News Event Retrieval Grand Challenge — Automatic Mode
- "AI autonomously retrieves and explains news events." Vietnamese news data. <https://acmmm2025.org/grand-challenge/>

### 6.3 MARS @ CASTLE/EgoVis 2026
- GPT-5.4 decision agent iteratively chooses: continue / request missing modality / answer / fallback. 0.57 accuracy on 185 questions × 15 perspectives × 4 days.
- **Pattern**: agent as evidence-router over heterogeneous modalities. [arXiv 2605.18176](https://arxiv.org/html/2605.18176v1)

### 6.4 Generic agentic-retrieval (2025-26)
- Cascaded retrieve-then-rerank with agent-guided decomposition ([arXiv 2512.12935](https://arxiv.org/pdf/2512.12935)): BEiT-3 + SigLIP recall in Qdrant; BLIP-2 cross-encoder rerank; GPT-4o decomposes query; temporal exponential-decay penalty.
- Smart routing for multimodal retrieval ([arXiv 2507.13374](https://arxiv.org/abs/2507.13374))
- VideoSeek (CVPR 2026): three tool primitives (scan / glance / zoom) using ~1/300 of frames. <https://github.com/jylins/videoseek>

### 6.5 Recommended autonomous architecture
```
User query (text or multimodal hint)
        ?
        ?
LLM Query Planner (GPT-4o / Qwen2.5-VL-7B-local)
  ??? parse {object, location, time, activity}
  ??? decompose into K sub-queries (visual, OCR, ASR)
  ??? emit execution plan: ranked tool calls + fusion
        ?
        ?
Tool registry execution (parallel)
  ??? tool-CLIP (OpenCLIP ViT-H/14)            ? Milvus dense kNN
  ??? tool-BLIP-2                              ? Milvus dense kNN
  ??? tool-OCR                                 ? Elastic BM25
  ??? tool-ASR (Whisper / PhoWhisper)          ? Elastic BM25
  ??? tool-ADL (Qwen-VL labels)                ? Milvus sparse
  ??? tool-Object/Color (canvas / detector)    ? Milvus sparse
        ?
        ?
Score normalization + RRF
        ?
        ?
Temporal coherence (SeqWin / ParChain w/ exponential decay)
        ?
        ?
Cross-encoder rerank (BLIP-2 or fine-tuned BERT)
        ?
        ?
Decision agent (GPT-4o):
  - confident? ? submit
  - low score? ? request missing modality, retry
  - max-iter? ? fallback heuristic
```

## 7. Master source list

**Reviews & result reports**
- LSC SOTA review: <https://arxiv.org/abs/2506.06743>
- VBS 2024 results: <https://arxiv.org/abs/2502.15683>
- VBS 2025 results: <https://arxiv.org/abs/2509.12000>
- VBS 2023 IEEE Access: <https://doi.org/10.1109/ACCESS.2024.3405638>
- LSC Archive: <https://github.com/lucaro/LSC-Archive>
- VBS Archive: <https://github.com/lucaro/VBS-Archive>

**System papers (LSC '24/'25)**
- SnapSeek 3.0: <https://dl.acm.org/doi/10.1145/3729459.3748697>
- MEMORIA '25: <https://dl.acm.org/doi/10.1145/3729459.3748693>
- MemoriEase 3.0: <https://doras.dcu.ie/31771/1/3729459.3748689.pdf>
- Memento 4.0: <https://doras.dcu.ie/30691/>
- MyEachtraX: <https://doras.dcu.ie/30727/1/3643489.3661128%20%281%29.pdf>
- LifeSeeker 6.0: <https://doras.dcu.ie/30559/>
- LifeGraph 5: <https://ceur-ws.org/Vol-4085/paper63.pdf>
- VISIONE LSC'24: <https://iris.cnr.it/handle/20.500.14243/485022>

**System papers (VBS '24/'25)**
- VISIONE 5.0: <https://zenodo.org/records/13903347> | code: <https://github.com/aimh-lab/visione>
- Vibro 2024: <https://files.visual-computing.com/research/Vibro_Video_Browsing_with_Semantic_and_Visual_Image_Embeddings.pdf>
- NII-UIT VBS'25: <https://doi.org/10.1007/978-981-96-2074-6_38>
- PraK V3 thesis: <https://dspace.cuni.cz/bitstream/handle/20.500.11956/202882/120518844.pdf>
- diveXplore CVPR'25: <https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Schoeffmann_AI-based_Video_Content_Understanding_for_Automatic_and_Interactive_Multimedia_Retrieval_CVPRW_2025_paper.pdf>
- VideoEase: <https://doras.dcu.ie/30858/1/MMM_VBS25_Linh.pdf>
- vitrivr-engine: <https://github.com/vitrivr/vitrivr-engine>
- MeGraS: <https://github.com/lucaro/MeGraS>

**Autonomous-agent**
- SnapMind: <https://doi.org/10.1007/978-981-95-6963-2_20>
- Cascaded multimodal agent: <https://arxiv.org/pdf/2512.12935>
- Smart routing: <https://arxiv.org/abs/2507.13374>
- ACMMM '25 Auto Mode: <https://acmmm2025.org/grand-challenge/>
- MARS: <https://arxiv.org/html/2605.18176v1>
- VideoSeek: <https://github.com/jylins/videoseek>

**Foundational models**
- CLIP2Video: <https://arxiv.org/abs/2106.11097>
- ALADIN: <https://arxiv.org/abs/2207.14757>
- BLIP-2: <https://arxiv.org/abs/2301.12597>
- BEiT-3: <https://arxiv.org/abs/2208.10442>
- EVA-CLIP: <https://arxiv.org/abs/2303.15389>
- TransNet V2: <https://doi.org/10.1145/3664647.3685517>
- DRES: <https://doi.org/10.1145/3678881>
- LSC-ADL: <https://arxiv.org/abs/2504.02060>
- OpenCLIP pretrained: <https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md>
