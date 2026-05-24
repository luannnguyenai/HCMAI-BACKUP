# Illustrations - for team discussion

> Four diagrams to anchor team discussions about the AIC2026 strategy and architecture. All AI-generated as conversation starters - the team should mark them up, debate the choices, and replace any inaccuracies before they ship to a slide deck.

## 1. System Architecture - `aic2026-system-architecture.png`

![System Architecture](aic2026-system-architecture.png)

What this shows: the full **offline indexing pipeline** (top) and **online query pipeline** (bottom) in one view.

Discussion prompts for the team:
- Do we agree on the three image encoders (SigLIP-2, Meta CLIP 2, InternVideo2-1B)? Should we replace InternVideo2 with V-JEPA-2 or LanguageBind?
- One Milvus, two Elasticsearch indexes - or fold everything into Milvus hybrid?
- Should the planner LLM also be in the offline pipeline (for caption generation)?
- Where does the audio-events (CLAP) lane fit? Add a fourth pipeline?

Reference: `docs/proposals/01-interactive-system-architecture.md`.

## 2. UI Mockup - `aic2026-ui-mockup.png`

![UI Mockup](aic2026-ui-mockup.png)

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

What this shows: the LangGraph state machine for the autonomous-agent track. Planner -> Tool Registry -> RRF -> Structured Filter -> VLM Critic -> Confidence gate -> Submit or Retry.

Discussion prompts for the team:
- 12 tools is what we sketched - are any missing (sound-event search via CLAP)? Should we drop generative_visual_query in MVP and add it in Phase 2?
- Confidence threshold 0.8 is a guess - how do we choose it? See `docs/proposals/05-evaluation-harness.md` SS 9 on Platt calibration.
- 3 retries: too many (wastes time budget) or too few (gives up on hard queries)? Maybe make it adaptive based on remaining wall-clock.
- Should the critic VLM also be allowed to *modify* the plan, not just score it (Reflexion-style)?

Reference: `docs/proposals/02-automatic-track-agent.md`.

## 4. Winning Hypothesis Stack - `aic2026-winning-stack.png`

![Winning Stack](aic2026-winning-stack.png)

What this shows: the **floor / edge / moat** framing of the winning hypothesis (regenerated May 24, 2026 after the strategy audit).

- **Tier 1 (Floor - bottom, muted slate-gray)**: reproduced 2026 SOTA stack. Meta CLIP 2 + SigLIP-2 + InternVideo2-1B + Vintern-3B + PhoWhisper + PaddleOCR/VietOCR + BGE-M3 in Milvus + Elasticsearch, DANTE for TRAKE, LLM planner + VLM-as-judge, speed UX. Tagged "necessary, not sufficient" — every serious 2026 team will land near this floor.
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
