# Illustrations - for team discussion

> Four diagrams to anchor team discussions about the AIC2026 strategy and architecture. All AI-generated as conversation starters - the team should mark them up, debate the choices, and replace any inaccuracies before they ship to a slide deck.

## 1. System Architecture - `aic2026-system-architecture.png`

![System Architecture](aic2026-system-architecture.png)

**STATUS: REGENERATED 2026-06-05.** Replaces the prior image, which predated three shipped decisions. The current image now reflects all of the previously-missing elements listed below:
- **Missing the Qwen3-VL-Embedding-2B offline visual-document lane** ([ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md), [SPEC-0004](../specs/SPEC-0004-image-embedding-service.md)/[SPEC-0006](../specs/SPEC-0006-milvus-schema-and-queries.md)): a 4th dense encoder, `encode_image`-only, indexed offline and fused via C2, never on the online query path.
- **Storage model**: the diagram draws "Milvus (our 3-encoder ensemble)" plus a separate baseline-CLIP Milvus. The shipped design ([SPEC-0006](../specs/SPEC-0006-milvus-schema-and-queries.md)) is a **single multi-vector `keyframes` collection** with named dense fields (siglip2 1152 / metaclip2 1024 / qwen3vl 2048) keyed by `pk = "<video_id>_<frame_id>"`.
- **No RTX 5070 online / GH200 offline hardware split** ([ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md)): image-tower extraction + all indexing run offline on GH200; only text-tower query encoding + planner + reranker run online on the 12 GB RTX 5070. This split is the spine of the current architecture and is absent here.
- **BGE-M3 dense+sparse heads** are not labelled in the OCR/ASR lanes (only "DiacriticBERT -> Elasticsearch" is shown).

Generation spec (applied 2026-06-05): a two-band diagram. **Offline band (label "OFFLINE - GH200 cloud burst"):** organiser-provided inputs (video, pre-extracted keyframes, object detection, weak baseline CLIP ViT-B/32 512-d, YouTube URLs in metadata) feeding: (a) four image encoders into ONE multi-vector Milvus `keyframes` collection - SigLIP-2 (1152-d), Meta CLIP 2 (1024-d), InternVideo2-1B (768-d), and a distinctly-styled **Qwen3-VL-Embedding-2B (2048-d) offline visual-document lane**; (b) the organisers' baseline CLIP kept as an ablation field in the same collection; (c) PaddleOCR PP-OCRv5 + VietOCR fallback -> Elasticsearch (OCR) with BGE-M3 dense+sparse + DiacriticBERT C1 MaxSim; (d) yt-dlp YouTube captions (primary, bright) + PhoWhisper-large/WhisperX (fallback, dashed) -> Elasticsearch (ASR) with BGE-M3 + DiacriticBERT C1. **Online band (label "ONLINE - RTX 5070, air-gapped"):** Vietnamese query -> Planner LLM (SeaLLMs-v3-7B INT4 local / Gemini 2.5 Flash escalation; auto-track variant distilled from operator traces, C4) -> parallel lanes (text-tower query encoding only: SigLIP-2 + Meta CLIP 2 text towers, OCR/ASR/caption retrieval) -> Per-task-type LambdaRank fusion (C2, RRF k=60 auto-fallback) -> structured filters -> DANTE DP for TRAKE -> Vintern-3B-beta VLM-as-judge rerank (INT4) -> top-10 -> React UI -> operator + submission-verification panel -> DRES; trace logger feeds C4. Use mint/emerald outlines for C1/C2/C4 (Edge, ADR-0007); a clear horizontal divider between the GH200 offline band and the 5070 online band; encoder dims labelled on each lane.

What this shows: the full **offline indexing pipeline** (top) and **online query pipeline** (bottom), reflecting the architecture as it stood after the dataset-shape intel ([research-note 06](../research-notes/06-aic2026-dataset-shape.md)) and the original-contributions decisions ([ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md), [ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md)).

Visual conventions:
- **Mint-green outlines** mark our three Edge contributions: C1 (DiacriticBERT late-interaction head, offline), C2 (per-task-type LambdaRank fusion, online), C4 (planner self-distillation from operator traces).
- **Dimmed gold** marks the organiser-provided lane (keyframes, OD, baseline CLIP, YouTube URLs in metadata). The baseline CLIP is kept only as an ablation lane; we don't expect it to beat our 3-encoder ensemble.
- **Brightness** in the ASR row reflects priority: yt-dlp YouTube captions are bright (primary), PhoWhisper-large is dimmed with a dashed border (fallback for videos lacking captions).

Discussion prompts for the team:
- Do we agree on the three image encoders (SigLIP-2, Meta CLIP 2, InternVideo2-1B)? Should we replace InternVideo2 with V-JEPA-2 or LanguageBind?
- One Milvus collection per encoder, or one Milvus collection with multiple vector fields? **RESOLVED ([SPEC-0006](../specs/SPEC-0006-milvus-schema-and-queries.md) Q-c): one multi-vector `keyframes` collection with named dense fields.** Do we keep the organisers' baseline CLIP indexed (gold lane) for ablation? See research-note 06 SS 2.3 for the rationale (deferred as a field in SPEC-0006 Q-b).
- yt-dlp primary, PhoWhisper fallback - what is our empirical caption-coverage threshold for dropping PhoWhisper entirely? (PhoWhisper is CC-BY-NC-SA; coverage-driven removal would close strategy SS 10 item 2.)
- DiacriticBERT is shown applied to both OCR and ASR text. Is there a case for applying it only to ASR (which is noisier) and not OCR? Defer to SPEC-0014.
- Where does the audio-events (CLAP) lane fit? Add a fifth pipeline if we end up using it; not currently in scope.
- The "auto-track variant: distilled from operator traces (C4)" annotation under the planner - is that the right hierarchy, or should the auto-track be drawn as a separate side-loop?

Reference: `docs/proposals/01-interactive-system-architecture.md`, `docs/proposals/08-original-contributions.md`, [`docs/research-notes/06-aic2026-dataset-shape.md`](../research-notes/06-aic2026-dataset-shape.md).

## 2. UI Mockup - `aic2026-ui-mockup.png`

![UI Mockup](aic2026-ui-mockup.png)

**STATUS: KEEP (reviewed 2026-06-05).** UI design is architecture-independent and still matches the React + WebSocket direction ([ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md)) and proposal 06; no drift.

What this shows: the React operator console in dark mode. Grid, planner JSON panel, query history, keyframe scrubber, frame-detail slideout, and the submission-verification bar.

Discussion prompts for the team:
- 8x4 grid vs 8x6 (more rows = smaller thumbnails, more context)?
- Is the Planner JSON panel a feature (operator sees the reasoning) or a distraction (clutter)?
- Where do we surface the Vietnamese ASR/OCR snippets for the selected frame - inline below thumbnail, in a tooltip, or only in the detail slideout?
- TRAKE mode is NOT shown here - we need a separate mockup for the 4-scene drag-drop staging tray.
- Is the verification bar countdown set correctly (3 seconds)? Different value for operator vs novice mode?

Reference: `docs/proposals/06-ui-ux-design.md`.

## 3. Automatic Track Agent Loop - `aic2026-agent-loop.png`

![Agent Loop](aic2026-agent-loop.png)

**STATUS: REGENERATED 2026-06-05.** The prior image drew the fusion node as "RRF Fusion (k=60)" as the default, which inverted the shipped decision: [ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md) makes **per-task-type learned fusion (C2) the default**, with RRF k=60 as the runtime auto-fallback. The current image now labels that node "Per-task-type LambdaRank Fusion (C2)" with an "RRF k=60 auto-fallback" sub-label, matching the system-architecture image.

Generation spec (applied 2026-06-05): keep the existing layout (Planner LLM -> Tool Registry of ~12 tools -> fusion -> Structured Filter -> Critic VLM-as-judge -> Confidence >= 0.8 gate -> DRES submit, with the "NO - retries <= 3" back-edge), but relabel the fusion node **"Per-task-type LambdaRank Fusion (C2)"** with a smaller sub-label **"RRF k=60 auto-fallback"**, styled with the same mint/emerald Edge outline used for C1/C2/C4 elsewhere. Everything else in the loop is still accurate (12-tool registry, confidence gate, 3-retry budget, per-query 30-60 s / < $0.02 budget).

What this shows: the LangGraph state machine for the autonomous-agent track. Planner -> Tool Registry -> per-task-type learned fusion (C2; RRF k=60 auto-fallback) -> Structured Filter -> VLM Critic -> Confidence gate -> Submit or Retry.

Discussion prompts for the team:
- 12 tools is what we sketched - are any missing (sound-event search via CLAP)? Should we drop generative_visual_query in MVP and add it in Phase 2?
- Confidence threshold 0.8 is a guess - how do we choose it? See `docs/proposals/05-evaluation-harness.md` SS 9 on Platt calibration.
- 3 retries: too many (wastes time budget) or too few (gives up on hard queries)? Maybe make it adaptive based on remaining wall-clock.
- Should the critic VLM also be allowed to *modify* the plan, not just score it (Reflexion-style)?

Reference: `docs/proposals/02-automatic-track-agent.md`.

## 4. Winning Hypothesis Stack - `aic2026-winning-stack.png`

![Winning Stack](aic2026-winning-stack.png)

**STATUS: KEEP (reviewed 2026-06-05).** Still accurate: the Floor/Edge/Moat tiers match the current strategy, C2 "replaces RRF k=60" matches [ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md), and the lift pills remain explicitly marked as Phase-2 estimates (correct - there is no ground-truth answer key for the AIC2025 proxy, so real recall@k / MRR are GT-blocked until the June-25 corpus). Optional future refresh: the Floor tier could add the Qwen3-VL-Embedding-2B offline visual-document lane ([ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md)); not required for accuracy.

What this shows: the **floor / edge / moat** framing of the winning hypothesis (regenerated May 24, 2026 after the strategy audit).

- **Tier 1 (Floor - bottom, muted slate-gray)**: reproduced 2026 SOTA stack. Meta CLIP 2 + SigLIP-2 + InternVideo2-1B + Vintern-3B + PhoWhisper + PaddleOCR/VietOCR + BGE-M3 in Milvus + Elasticsearch, DANTE for TRAKE, LLM planner + VLM-as-judge, speed UX. Tagged "necessary, not sufficient" - every serious 2026 team will land near this floor.
- **Tier 2 (Edge - middle, three vibrant emerald bars)**: our three original contributions. C1 DiacriticBERT (+2-5% R@1 est. on OCR/ASR slice), C2 Per-task Learned Fusion (+1-3% NDCG@10 est., replaces RRF k=60), C4 Agent Self-Distillation (+10-20% R@1 est. on automatic track via DSPy on operator traces).
- **Tier 3 (Moat - top, warm copper)**: operator drills + verification panel. Process advantage, not technology. PraK1 vs PraK2 = 30 pts on the same engine.
- **Podium base**: AIC2026 1st Prize, Bảng A.

Lift numbers are explicitly marked as Phase-2 ablation estimates in the bottom-right footnote. They become real once `eval/diacritic_robustness.py`, `eval/fusion_ablation.py`, and `eval/agent_automatic_ablation.py` (proposal 05 SS 13-15) start producing nightly numbers in Phase 2.

Discussion prompts for the team:
- Replace the estimated lift pills with real numbers from the eval harness once Phase 2 ablations have at least 3 stable nightly runs.
- If any of {C1, C2, C4} fails its ship gate, redraw that bar in muted gray with the "shipped fallback" label (RRF for C2, SeaLLMs-v3 query-rewrite for C1, zero-shot planner for C4) so the slide stays honest.
- This is the slide we present to organisers / press if we win. The visual hierarchy is deliberate: Tier 1 is large but muted ("everyone has this"); Tier 2 is mid-size and vibrant ("what makes us win"); Tier 3 is small but distinct ("process advantage compounding edge into score").

Reference: `docs/strategy/00-master-strategy.md` SS 2 and `docs/proposals/08-original-contributions.md`.

---

## How these were made

Generated via AI image-gen tool on 2026-05-24 from detailed prompts informed by the strategy and proposals docs. They are **starting points for team discussion**, not final assets. As decisions firm up, replace these with Figma-authored versions for the press kit / slide deck.

Open them in your image viewer at 100% to read the labels.
