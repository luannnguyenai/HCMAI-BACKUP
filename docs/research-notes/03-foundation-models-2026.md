# Research Note 03 — Foundation Models for Multimedia Retrieval (2026 State-of-the-Art)

> Compiled May 2026. Every claim is anchored to an arXiv paper, HF card, or GitHub repo. Use this as the "what model do I pick for which subsystem" reference.

---

## A. Image–Text Dual Encoders (text ? image retrieval at scale)

### A.1 SigLIP 2 (Google, Feb 2025) — DEFAULT BACKBONE
- Multilingual VLM with captioning pretraining + self-distillation + masked prediction + online curation on top of sigmoid loss.
- Sizes: ViT-B/16 (86M), L/16 (303M), So400m/16 (400M), g-opt/16 (1B). NaFlex variants preserve native aspect ratio.
- Benchmarks: ImageNet-1k zero-shot — B 74.0 / L 78.2 / So400m@384 84.1 / **g-opt@384 85.0**. COCO I?T R@1: L/16@512 72.1, g-opt@384 72.8. Strong on XM3600 (36 languages incl. Vietnamese).
- HF: [`google/siglip2-so400m-patch16-512`](https://huggingface.co/google/siglip2-so400m-patch16-512), [`google/siglip2-giant-opt-patch16-384`](https://huggingface.co/google/siglip2-giant-opt-patch16-384). License: Apache-2.0.
- VRAM: L/16@512 <8 GB FP16; So400m@384 ~10 GB; g-opt@384 ~18 GB.
- Paper: <https://arxiv.org/abs/2502.14786>

### A.2 Meta CLIP 2 (Meta FAIR, NeurIPS 2025) — BEST MULTILINGUAL / VIETNAMESE
- First CLIP from scratch on worldwide web-scale data without "curse of multilinguality."
- ViT-H/14 surpasses English-only MetaCLIP by +0.8 on ImageNet and sets multilingual SOTA: CVQA 57.4, Babel-ImageNet 50.2, **XM3600 I?T 64.3** (best non-English in 2025).
- Beats SigLIP 2 on Babel-IN +3.8, XM3600 +1.1/+1.5, CVQA +3/+7.6, Flickr-30k-200 +7.7/+7, XTD-200 +6.4/+5.8.
- HF: [`facebook/metaclip-2-worldwide-huge-quickgelu`](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu). License: MIT.
- Paper: <https://arxiv.org/html/2507.22062v3> | Repo: <https://github.com/facebookresearch/MetaCLIP>
- **Verdict**: If Vietnamese queries dominate, this is **the** dual encoder.

### A.3 EVA-02-CLIP (BAAI)
- EVA-02-CLIP-L/14+ (428M): 80.4% IN-1k zero-shot.
- **EVA-02-CLIP-E/14+ (5.0B)**: **82.0% IN-1k** — top open ViT-E for years.
- License MIT; English-only training.
- Repo: <https://github.com/baaivision/EVA/tree/master/EVA-CLIP>
- **Use as**: strong English second opinion / reranker.

### A.4 OpenCLIP ViT-H/14 DFN-5B
- 84.37% IN-1k, 70.79 avg on 38 datasets. 990M params, Apache-2.0.
- Repo: <https://github.com/mlfoundations/open_clip>
- **Use as**: stable, well-tooled fallback.

### A.5 Long-CLIP (ECCV 2024)
- Extends CLIP text length 77 ? 248 tokens, drop-in compatible.
- +20% R@5 on long-caption retrieval, +6% standard.
- Repo: <https://github.com/beichenzbc/Long-CLIP>
- **Use for**: paragraph-length queries (TRAKE's 4-scene format).

### A.6 DreamLIP (ECCV 2024)
- MLLM-generated long captions + sub-caption sampling + grouping loss. With only 30M pairs matches/exceeds CLIP-400M on retrieval.
- Repo: <https://github.com/ant-research/DreamLIP>
- **Use as**: recipe template for "synthetic long-caption augmentation" fine-tuning.

### A.7 PE-Core-bigG-14-448 (Meta) — Vietnamese-news domain ?
- 85.4% IN-1k, 58.1% COCO t2i, 51.2% VATEX — above CLIP ViT-H/14 and OpenCLIP bigG/14.
- Used by **MERVIN** (HCMC AIC 2025 system) as the primary encoder.
- Reference: [MERVIN arXiv 2605.16120](https://arxiv.org/html/2605.16120v1)

### A.8 BLIP-2 ITC
- Salesforce; Q-Former ITC head. Superseded for retrieval by SigLIP-2/Meta CLIP 2 but **excellent for ITM-style reranking**.

### A.9 Qwen3-VL-Embedding (Alibaba, Jan 2026) - screened SPEC-0025, deferred as online encoder
- Unified multimodal embedder (text+image+video -> one space), Apache-2.0, 2B (2048-d) / 8B (4096-d), MRL, **MMEB-V2 SOTA** (8B 77.8; strong visual-document retrieval). Official API: the QwenLM/Qwen3-VL-Embedding repo's `Qwen3VLEmbedder.process()` (instruction-aware), **not** plain `transformers.AutoModel`.
- **Bake-off vs the floor on the AIC2025 proxy (SPEC-0025, 2026-06-03):** query-encode latency **~5x** SigLIP-2/Meta CLIP 2 (52.7 ms vs ~11-12 ms p50 on H200); and being a *unified 2B* it has no lightweight online text tower (unlike the CLIP-style floor), so its online footprint is the full model (~1.5-2 GB INT4, tight in the 5070's ~3 GB headroom; ADR-0003). **Verdict: do not adopt as the online query encoder; likely value is an offline visual-document lane (8B, fused), pending rigorous R@k (no 2025 ground truth).**
- **Gemini Embedding 2** is disqualified for the retrieval path entirely: closed-weight API, incompatible with the air-gapped 5070 finals (the query encoder must run locally).

### Summary table (image-text)

| Model | Params | IN-1k 0-shot | XM3600 | License | Best for |
|---|---:|---:|---|---|---|
| **SigLIP 2 g-opt/16@384** | 1B | **85.0** | strong | Apache-2.0 | Default backbone |
| **Meta CLIP 2 H/14** | ~1B | 81.3 | **64.3 I?T** | MIT | Vietnamese queries |
| EVA-02-CLIP-E/14+ | 5.0B | 82.0 | EN-only | MIT | English ImageNet rerank |
| OpenCLIP H/14 DFN-5B | 990M | 84.4 | EN-only | Apache-2.0 | Stable infra |
| Long-CLIP-L | 430M | ~75 | EN-only | MIT | TRAKE long queries |
| PE-Core-bigG-14-448 | ~2B | 85.4 | (used VN) | TBD | HCMC news domain |

---

## B. Video Understanding / Clip Embedding

### B.1 InternVideo2 — TOP VIDEO PICK
- Multi-stage (mask recon ? video-text contrastive ? instruction tune). 1B / 6B.
- Zero-shot MSR-VTT T2V R@1: **51.9 (1B), 55.9 (6B)**; R@5 75.3 / 78.3.
- HF: [`OpenGVLab/InternVideo2-Stage2_1B-224p-f4`](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4). Apache-2.0.
- 1B fits in 6 GB FP16 (4 frames); 6B ~24 GB. Doubles as image CLIP.

### B.2 V-JEPA 2 (Meta FAIR, June 2025)
- Self-supervised joint-embedding predictive; predicts masked latent features.
- Sizes ViT-L (300M), H (600M), g (1B), g-384 (1B).
- Frozen-probe scores: SSv2 77.3, Diving48 90.2, EK-100 anticipation 39.7. Best motion understanding.
- **Caveat**: no video-text contrastive head ? must train one. MSR-VTT only 34.4 with MobileCLIP text alignment.
- License: non-commercial research.
- Repo: <https://github.com/facebookresearch/vjepa2>

### B.3 LanguageBind (PKU)
- Aligns video / audio / depth / thermal / image to a frozen language space.
- MSR-VTT T2V R@1: 42.8 (L), 44.8 (H). MIT.
- **Use as**: cheap multimodal baseline.

### B.4 VideoMAE V2
- 1B-params masked-autoencoder. SOTA action recognition but no native text alignment.

### B.5 Qwen2.5-VL / Qwen3-VL (video mode) — BEST END-TO-END
- Native video input with absolute time encoding, up to ~1 hour. Used as encoder + reasoner.
- Apache-2.0 for most sizes.

### B.6 VideoLLaMA 3 (DAMO/Alibaba, 2025)
- 7B/72B video-language. Apache-2.0. Use for captioning + dense description.

### B.7 CLIP4Clip / CLIP2Video
- CLIP4Clip (2021): cheap fine-tuning baseline (mean-pool / transformer / tight-transformer scoring on frozen CLIP).
- CLIP2Video: TDB+TAB blocks, used by VISIONE.

### Summary table (video)

| Model | Params | MSR-VTT R@1 | Strengths | License |
|---|---:|---:|---|---|
| **InternVideo2-6B** | 6B | **55.9** | Best retrieval | Apache-2.0 |
| InternVideo2-1B | 1B | 51.9 | Fits 4090 | Apache-2.0 |
| LanguageBind-H | 1B | 44.8 | Multi-modal hub | MIT |
| V-JEPA 2-g | 1B | 34.4* | SOTA action probes | NC research |
| Qwen2.5-VL-7B (video) | 7B | — (gen) | VQA + caption | Apache-2.0 |

---

## C. Multimodal LLMs (VLMs)

### Closed-weight frontier (API)

| Model | Strength | Pricing $/M | Vision | Context |
|---|---|---|---|---|
| **Gemini 2.5 Pro / 3 Flash / 3.1 Pro** | Best native video; 1M ctx | 0.50–1.25 / 3–10 | MMMU-Pro 81–84 | 1M |
| GPT-5.x (5, 5.1, 5.2) | Strong vision + reasoning | varies | MMMU-Pro 75 | 256K |
| Claude 4.5 Sonnet / Opus 4.5 | Best code, strong vision | varies | MMMU-Pro 74 | 200K |

**Gemini 2.5/3 Flash is the recommended VLM-as-judge for reranking** — 1M ctx fits the whole top-30 + cot rationale; cost = $0.50/M; native video.

### Open-weight VLMs

#### Qwen2.5-VL / Qwen3-VL (Alibaba) — TOP OPEN PICK
- 3B / 7B / 32B / 72B + AWQ INT4. Native dynamic resolution + video timestamps + grounding boxes.
- Qwen3-VL adds MoE up to 235B-A22B, "Thinking" reasoning edition; **best self-hostable VLM** (MMMU-Pro 69% on 235B).
- VRAM: 7B BF16 13 GB, 72B INT4 33 GB.
- Apache-2.0 (most sizes).
- Repos: <https://github.com/QwenLM/Qwen2.5-VL>, <https://github.com/QwenLM/Qwen3-VL>

#### InternVL 3 / 3.5 (OpenGVLab) — STRONGEST OPEN MMMU
- InternVL 3.5 (Aug 2025): 1B ? 241B-A28B. **MMMU 73.4 (8B), 77.7 (241B)** — 8B beats most 70B-class models.
- HF: [`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B). MIT.
- **Use for**: yes/no rerank probes.

#### Vintern-1B/3B (5CD-AI) — VIETNAMESE-NATIVE VLM ?
- 5CD-AI/Vintern-3B-beta: #2 on MTVQA Vietnamese, beats GPT-4o on Vi tasks.
- Vintern-1B-v2/v3.5: small, fast on Vietnamese.
- **ColVintern-1B-v1**: ColBERT-style late-interaction Vietnamese document retriever — UNIQUE multilingual ColPali equivalent.
- HF: <https://huggingface.co/5CD-AI>
- **Use for**: VLM-as-judge reranker on Vietnamese queries.

#### Pixtral 12B (Mistral, Sept 2024)
- 12B Apache-2.0. MMMU 52.5, DocVQA 90.7. Single-A6000 deploy.

#### DeepSeek-VL2 (DeepSeek, Dec 2024)
- 1B / 2.8B / 4.5B MoE. OCR/table strong. MIT.

#### MiniCPM-V 2.6 / 3.0 (OpenBMB)
- 8B, top of MTVQA, mobile-runnable. Mixed licenses.

#### Phi-4 Vision (Microsoft)
- 5.6B compact, MIT. Math/diagram reasoning at low cost.

#### Molmo 72B (Allen AI)
- Apache-2.0, point-and-grounding focused.

#### Llama 3.2-Vision / Llama 4
- 11B/90B. Pixtral 12B beats it by ~20pt on MMMU. Llama 4 unique 10M ctx.

#### PaliGemma 2 / 3 (Google)
- 3B/10B/28B. Recommended by Google as "fine-tune-first" base. Gemma license.

### Summary (VLMs)

| Model | Params (active) | MMMU(-Pro) | Context | License | VRAM |
|---|---:|---:|---|---|---|
| **Gemini 3 Flash (API)** | proprietary | 81.2% Pro | 1M | Closed | $0.50/$3 per M |
| **Qwen2.5-VL-72B INT4** | 72B | ~70 | 1M (YaRN) | Apache-2.0 | 33 GB |
| Qwen2.5-VL-7B BF16 | 7B | ~58 | 128K | Apache-2.0 | 13 GB |
| InternVL 3.5-8B | 8B | **73.4** | 128K | MIT | 16 GB |
| Vintern-3B-beta | 3B | (#2 MTVQA Vi) | 8K | Apache-2.0 | 7 GB |
| Pixtral 12B | 12B | 52.5 | 128K | Apache-2.0 | 24 GB |

---

## D. OCR / Scene-Text

### D.1 PaddleOCR PP-OCRv5 + PaddleOCR-VL — TOP OCR
- PP-OCRv5 (0.07B) covers **111 languages including Vietnamese**.
- **Outperforms Qwen2.5-VL-72B, InternVL3-78B, Gemini-2.5 Pro, GPT-4o, GOT-OCR2.0** in average 1-edit-distance.
- PaddleOCR-VL (1.3B): 94.5% on OmniDocBench v1.5; layout + text + formula + table + reading order.
- Apache-2.0.
- Repo: <https://github.com/PaddlePaddle/PaddleOCR>
- Paper: <https://arxiv.org/html/2507.05595>

### D.2 VietOCR (pbcquoc) — VIETNAMESE-SPECIFIC
- Transformer/Seq2Seq with VGG19-bn backbone. 0.880 full-sequence precision on 10M-image internal; 0.890/0.981 (full-seq/per-char) on MCOCR2021 receipts.
- Apache-2.0.
- Repo: <https://github.com/pbcquoc/vietocr>
- **Workflow**: PaddleOCR detection + VietOCR recognition via [`kaylode/vnm-ocr-toolbox`](https://github.com/kaylode/vnm-ocr-toolbox).

### D.3 GOT-OCR 2.0 (StepFun)
- 580M unified OCR-2.0. Markdown/LaTeX output. HF: <https://huggingface.co/stepfun-ai/GOT-OCR2_0>

### D.4 Surya
- Modern multilingual (90+ langs, Vietnamese). <2 GB VRAM. Apache-2.0. <https://github.com/VikParuchuri/surya>

### D.5 EasyOCR / DocTR / Tesseract
- EasyOCR: 80+ langs incl. VN, easiest install, lower accuracy than PaddleOCR.
- DocTR: Apache-2.0, modular detection+rec.
- Tesseract: legacy fallback.

### Recommended OCR pipeline
```
keyframe ? PP-DocLayoutV3 (layout)
        ?? PP-OCRv5 latin (Vietnamese rec) ? text + boxes
              ?? fallback: VietOCR for handwritten
              ?? PaddleOCR-VL or Qwen2.5-VL-7B for tables/formulas
```

---

## E. ASR / Speech-to-Text (Vietnamese)

### E.1 PhoWhisper (VinAI, ICLR 2024) — TOP VIETNAMESE ?
- Whisper fine-tuned on 844 hours of Vietnamese.
- **PhoWhisper-large (1.55B)** WER on Vietnamese benchmarks: CMV-Vi **8.14**, VIVOS **4.67**, VLSP-2020 Task-1 **13.75**, Task-2 **26.68** — SOTA.
- HF: [`vinai/PhoWhisper-large`](https://huggingface.co/vinai/PhoWhisper-large) + tiny/base/small/medium.
- **License: CC-BY-NC-SA-4.0** — verify with AIC organisers; research/non-commercial only.
- Paper: <https://arxiv.org/pdf/2406.02555>

### E.2 Whisper large-v3 / large-v3-turbo (OpenAI)
- Turbo: 4 decoder layers, 7-10× faster, ~9.5% WER on standard English.
- HF: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo). MIT.
- **Vietnamese**: decent with code-switch; weaker than PhoWhisper on Vi-only.

### E.3 Faster-Whisper / WhisperX
- Faster-Whisper: CTranslate2 reimpl, 2-4× faster, half VRAM. <https://github.com/SYSTRAN/faster-whisper>
- WhisperX: Faster-Whisper + forced alignment + Pyannote diarization ? word-level timestamps. <https://github.com/m-bain/whisperX>

### E.4 NVIDIA Canary-1B-v2 / Parakeet-TDT-0.6B-v3 (Sept 2025)
- Canary-1B-v2: 25 European langs, avg 7.15% WER, RTFx 749 (A100). CC-BY-4.0.
- Parakeet-TDT-0.6B-v3: RTFx 3332 — fastest on the leaderboard, 6.32% avg WER.
- **Vietnamese NOT supported** in v2 — use only for English code-switched portions.

### E.5 wav2vec2-large-vi-vlsp2020 (VinAI / VietAI)
- 317M. WER 8.61 VIVOS, 36.75 VLSP-T2 — beaten by PhoWhisper-medium.

### E.6 Pyannote Audio 3.x
- Diarization SOTA, ~7% DER on AMI. <https://huggingface.co/pyannote>. MIT.

### Recommended ASR
```
audio ? PhoWhisper-large (Vietnamese) ? text + WhisperX timestamps
     ?? Whisper-large-v3-turbo (English/code-switch)
     ?? Pyannote 3.1 (diarization)
```

---

## F. Audio Embeddings / Sound Events

### F.1 LAION-CLAP — TOP AUDIO RECOMMENDATION
- HTSAT/PANN audio + CLIP/Roberta text. Best ckpts:
  - `630k-audioset-fusion-best.pt` — variable-length general audio.
  - `music_audioset_epoch_15_esc_90.14.pt` — **ESC-50 zero-shot 90.14%**.
  - `music_speech_audioset_epoch_15_esc_89.98.pt` — music+speech+general combo.
- CC0 weights, MIT code. <https://github.com/LAION-AI/CLAP>

### F.2 Microsoft CLAP 2023
- HTSAT-22 + GPT-2 encoders. AudioCaps T?A R@1 0.51, mAP@10 0.257. MIT.

### F.3 BEATs (Microsoft)
- SOTA AudioSet 50.6% mAP. Feature extractor + small classification head.

### F.4 Wav2CLIP / SONAR / AST / AudioMAE
- Wav2CLIP: aligns audio to frozen CLIP image embeddings ? free text-audio retrieval.
- SONAR: 200-lang multimodal sentence-level encoder.

### Recommended audio
```
ambient ? LAION-CLAP music_speech_audioset ? 512-d ? FAISS
speech ? PhoWhisper text ? BGE-M3 ? text FAISS
```

---

## G. Cross-Encoder / Re-rankers

### G.1 BLIP-2 ITM head (classic)
- `Salesforce/blip2-itm-vit-g`; single match score per (image, caption) pair. BSD-3.

### G.2 EVA-CLIP-E + light cross-attention (CoCa-style)
- Reuses cached EVA features.

### G.3 VLM-as-reranker (2026 DEFAULT)
- Send query + 3×3 grid of top-9 candidates to Qwen2.5-VL-7B or **Gemini 2.5 Flash**:
> "Rank these images 1–9 by relevance to: <query>. Think step by step."
- Recipe:
  1. Top-200 from SigLIP-2 (dense)
  2. Top-50 from Meta CLIP 2 (Vietnamese dense)
  3. RRF ? top-30
  4. VLM rerank top-30 with CoT ? final top-10

### G.4 InternVL 3.5 yes/no probe
- 8B reasoning edition runs "is candidate i relevant?" with logits. MIT, fast on 4090 at 8-bit.

### G.5 ColPali / ColQwen / ColVintern
- Late-interaction with VLM tokens. Better as **first-stage** retriever than reranker.

---

## H. Dense Text Retrievers + Sparse / Hybrid

### H.1 BGE-M3 (BAAI) — TOP MULTILINGUAL
- 170+ languages, 1024-d dense + sparse + ColBERT multi-vector in one model. 8192-token context.
- Strong on Vietnamese.
- MIT. <https://huggingface.co/BAAI/bge-m3>

### H.2 multilingual-e5-large-instruct (Microsoft, MMTEB 2025)
- XLM-R-large (560M) + instructions. **Best sub-1B on MMTEB**. MIT.

### H.3 jina-embeddings-v3 (2024)
- 570M, 89 langs incl. VN, 8192 tokens, task-specific LoRA adapters, Matryoshka 32-1024d.
- MTEB EN avg 65.52 — beats text-embedding-3-large.
- **License: CC-BY-NC-4.0** — research only.

### H.4 jina-embeddings-v4 (2025)
- 3.8B, multimodal (text + image + PDF). Apache-2.0 weights, proprietary API. ViDoRe 90.17, MMTEB 66.49.

### H.5 mGTE (Alibaba)
- gte-multilingual-base: native 8192 ctx, RoPE, matches BGE-M3 at half params. Apache-2.0.

### H.6 Cohere Embed v4
- Closed API. Text + images + PDFs. 128K ctx, 1536-d Matryoshka. 100+ langs.

### H.7 Qwen3-Embedding
- 0.6B/4B/8B Apache-2.0; 8B MTEB EN 70+.

### H.8 Vietnamese-specific
- **PhoBERT-large** (VinAI): 370M, Vi RoBERTa baseline. <https://huggingface.co/vinai/phobert-large>
- **bkai-foundation-models/vietnamese-bi-encoder**: SimCSE Vi-only retriever. MIT.
- **dangvantuan/vietnamese-embedding** (used by MERVIN).

### H.9 BM25
- Python: `rank_bm25`. Production: **Pyserini** (best, JVM, used in ViRE).
- Vietnamese tokenization: **pyvi**, **underthesea**, **VnCoreNLP** before BM25.

### H.10 SPLADE-v3 (Naver)
- 30,522-d sparse over BERT vocab. MRR@10 40.2 on MS-MARCO, +2 nDCG@10 on BEIR-13 vs SPLADE++. CC-BY-NC-SA, English. Train Vi version on competition data.

### H.11 ColBERT-v2 + Vietnamese variants
- Late-interaction MaxSim. ~50× cheaper than cross-encoder, ~2× more accurate than dense.
- Use BGE-M3's ColBERT head out of the box, or train ColBERT on PhoBERT.

### Recommended hybrid
```
text query ? BGE-M3 dense ??
         ?? BGE-M3 sparse ???? RRF (k=60) ? top-50 ? VLM rerank
         ?? BM25 (Pyserini)?
```

> **Note for our system:** this is the documented 2026 best-practice hybrid. Our chosen architecture (proposal 01 SS 5.11) extends it in two places: (a) DiacriticBERT MaxSim (C1) is added as a fourth ranker for Vietnamese OCR/ASR text; (b) RRF k=60 is replaced by a per-task-type LightGBM LambdaRank (C2) with RRF as the runtime auto-fallback. See [`../proposals/08-original-contributions.md`](../proposals/08-original-contributions.md) for the rationale and ablation plan.

---

## I. Fine-tuning for Retrieval

### I.1 LoRA on CLIP/SigLIP — 0.5-1% params, fits 4090 even for SigLIP-2-g
- Add rank-8/16 LoRA on QKV of both towers; contrastive loss on in-domain data.
- Tooling: PEFT + open_clip + dataloader.

### I.2 Prompt tuning / CoOp / MaPLe — <0.01% params, no arch change
- 8-16 soft prompts; great when data is small.

### I.3 Distillation
- 5B EVA-02-CLIP-E ? 300M student ? re-encode 1M frames in hours.

### I.4 Synthetic data with LLMs (2026 DEFAULT)
- Use Gemini 2.5 / Qwen2.5-VL-72B to re-caption keyframes with **long detailed Vietnamese captions** (DreamLIP recipe). Contrastive-train CLIP on those pairs.
- Bonus: paraphrased Vietnamese variants for query augmentation at test time.

### I.5 ALBEF-style hard negatives
- Sample hard negs from top-k of an existing model; refine with InfoNCE + ITM.

### I.6 ColPali (Illuin Tech)
- Late-interaction VL retriever on PaliGemma; multi-vector page embeddings via MaxSim.
- SOTA on ViDoRe (incl. Vietnamese pages).
- Repo: <https://github.com/illuin-tech/vidore-benchmark>
- HF: <https://huggingface.co/vidore/colpali-v1.3>
- **Use as**: page-level retriever for slide/document-heavy frames.

### I.7 ColQwen2 / ColInternVL / ColVintern
- ColPali-style adapters on Qwen2-VL / InternVL / **Vintern (Vietnamese)**.
- ColVintern is unique — the only Vietnamese ColBERT-style document retriever.

### I.8 VisRAG (THUNLP)
- Encode page-image with VLM; retrieve via dense vector; generation step uses original image, avoiding OCR loss.
- Repo: <https://github.com/openbmb/visrag>

### 3-week fine-tuning sprint
1. **W1**: SigLIP-2-So400m + BGE-M3 + PhoWhisper + PP-OCRv5 baseline, no fine-tune.
2. **W2**: Re-caption keyframes with Qwen2.5-VL-72B-INT4 in Vietnamese. LoRA-fine-tune SigLIP-2-L on synthetic pairs.
3. **W3**: Train ColVintern late-interaction on competition-similar data; add Gemini 2.5 Flash reranker on top-30.

---

## J. Recommended 2026 stack (1×A6000 / 4090 + cloud burst)

| Subsystem | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| Image-text dual encoder | **SigLIP-2 So400m/16@384** | **Meta CLIP 2 H/14** (Vietnamese) | EVA-02-CLIP-L/14+ |
| Long-caption variant | Long-CLIP-L | DreamLIP ViT-L | SigLIP-2 sliding |
| Video encoder | InternVideo2-1B | InternVideo2-6B (cloud) | LanguageBind-L |
| VLM reranker (closed) | **Gemini 2.5 Flash** | GPT-5.2 | Claude 4.5 Sonnet |
| VLM reranker (open) | **Qwen2.5-VL-7B** | Qwen2.5-VL-72B INT4 | InternVL 3.5-8B |
| Vietnamese VLM | **Vintern-3B-beta** | Vintern-1B-v3.5 | Qwen2.5-VL-Vi |
| OCR | **PaddleOCR PP-OCRv5** | VietOCR (handwritten) | PaddleOCR-VL |
| ASR Vietnamese | **PhoWhisper-large** + WhisperX | Whisper-large-v3-turbo | wav2vec2-large-vi |
| Diarization | Pyannote-audio 3.1 | WhisperX | NeMo |
| Audio events | LAION-CLAP music_speech_audioset | MS-CLAP 2023 | BEATs + probe |
| Multilingual text retriever | **BGE-M3** | mE5-large-instruct | jina-v3 |
| Vietnamese-only text | bkai-vietnamese-bi-encoder | PhoBERT-large fine-tuned | XLM-R-large |
| Sparse / lexical | BM25 Pyserini + pyvi | SPLADE-v3 (EN) | MongoDB Atlas Fuzzy |
| Vector index | **FAISS IVF-PQ** for memory or **Milvus** for hybrid | Qdrant | Vespa |
| Visual document RAG | ColVintern-1B / ColPali | ColQwen2 | VisRAG |
| Reranker | **Gemini 2.5 Flash VLM rerank top-30** | InternVL 3.5-8B yes/no | BLIP-2 ITM head |

### Cold-indexing cost on 1×A6000 48 GB (1M frames)

| Model | GPU hours | VRAM at inference |
|---|---:|---:|
| SigLIP-2 So400m@384 | ~12 | 8 GB |
| InternVideo2-1B (4 frames/clip) | ~24 | 6 GB |
| PP-OCRv5 (CPU OK) | ~6 | <2 GB |
| PhoWhisper-large + WhisperX | ~4 (per 100h audio) | 6 GB |
| BGE-M3 (dense+sparse) | ~3 | 4 GB |
| LAION-CLAP | ~2 | 2 GB |
| **Cold indexing total** | **~50 GPU hours** | — |

### Online latency targets
- p50: <800 ms
- p95: <2 s

---

## Sources (selected)
- SigLIP 2 paper: <https://arxiv.org/abs/2502.14786>
- Meta CLIP 2: <https://arxiv.org/html/2507.22062v3>
- EVA-CLIP: <https://github.com/baaivision/EVA/tree/master/EVA-CLIP>
- open_clip: <https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md>
- Long-CLIP: <https://github.com/beichenzbc/Long-CLIP>
- DreamLIP: <https://arxiv.org/html/2403.17007>
- InternVideo2 benchmarks: <https://opencodepapers-b7572d.gitlab.io/benchmarks/zero-shot-video-retrieval-on-msr-vtt.html>
- V-JEPA 2: <https://arxiv.org/html/2506.09985>
- Qwen2.5-VL: <https://github.com/QwenLM/Qwen2.5-VL>
- Qwen3-VL: <https://github.com/QwenLM/Qwen3-VL>
- InternVL 3.5: <https://arxiv.org/html/2508.18265v2>
- PaddleOCR 3.0: <https://arxiv.org/html/2507.05595>
- VietOCR: <https://github.com/pbcquoc/vietocr>
- PhoWhisper: <https://arxiv.org/pdf/2406.02555>
- Faster-Whisper: <https://github.com/SYSTRAN/faster-whisper>
- LAION-CLAP: <https://github.com/LAION-AI/CLAP>
- BGE-M3: <https://huggingface.co/BAAI/bge-m3>
- MMTEB ICLR 2025: <https://proceedings.iclr.cc/paper_files/paper/2025/file/fc0e3f908a2116ba529ad0a1530a3675-Paper-Conference.pdf>
- jina-v3: <https://arxiv.org/html/2409.10173v1>
- SPLADE-v3: <https://arxiv.org/html/2403.06789v1>
- ColPali: <https://huggingface.co/papers/2407.01449>
- Vintern (5CD-AI): <https://huggingface.co/5CD-AI>
- HCMC AIC 2025 EEIoT_newbie paper: <https://arxiv.org/html/2512.06334>
- MERVIN (PE-Core + Vietnamese): <https://arxiv.org/html/2605.16120v1>
