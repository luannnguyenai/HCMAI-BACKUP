# Proposal 08 - Original Contributions on top of the 2026 SOTA Stack

> Proposals 01-07 describe an integration of best-of-breed 2026 SOTA models and patterns drawn from prior LSC/VBS/AIC winners (MEMORIA, NII-UIT, SnapMind, AIO_Owlgorithms). That integration is necessary but not sufficient: any team that reads the same LSC SOTA review and the same MMM 2026 SnapMind paper will land on ~90% of the same stack. This proposal defines **what is novel about our system** - five contributions designed to be small enough to ship inside the 17-week timeline and large enough to be defensible in a finals Q&A.

## 1. Why this proposal exists

An audit of `README.md` and proposals 01-07 against the question *"Does this propose a new method, or just integrate an existing one?"* found that **23 of ~25 components are off-the-shelf reuse**, and that the four "winning advantages" in the master strategy are each explicitly attributed to a prior team. As a competition plan, that is fine - integration quality and operator training have historically determined LSC/VBS winners. As an *impressive* technical strategy, it is not differentiated.

This proposal commits us to three first-class novelty workstreams (`C1`, `C2`, `C4`) and two backup workstreams (`C3`, `C5`). Each is scoped so it can be ablated on a 200-query dev set, defended in a slide, and - if it works - written up as a short paper after the competition.

## 2. The five contributions at a glance

| ID | Name | Problem it solves | Effort | Expected lift | Status |
|---|---|---|---|---|---|
| **C1** | **DiacriticBERT**: diacritic-robust Vietnamese bi-encoder head | Vietnamese ASR/OCR systematically strip or mangle diacritics; off-the-shelf BGE-M3 does not model the noise distribution | ~1 week (1 engineer) | +2-5% R@1 on OCR/ASR-bridged queries | **Primary** |
| **C2** | **Learned per-task-type fusion** replacing uniform RRF k=60 | We stack 12-15 ranked lists from very different score distributions; uniform RRF is provably suboptimal | ~3 days | +1-3% R@5 on dev set; bigger on Ad-hoc | **Primary** |
| **C3** | **PriorDP**: story-graph generalisation of DANTE for TRAKE | DANTE's linear lambda penalty ignores learnable scene-transition priors observable in the corpus | ~2 weeks | +3-7% TRAKE accuracy | Backup (ship if Phase 2 has slack) |
| **C4** | **Agent self-distillation** from interactive operator traces | Automatic track is new in 2026; no prior team has published a planner trained on its own operator's tool-choice traces | ~1 week (after interactive system stabilises) | +10-20% automatic-track score vs zero-shot planner | **Primary** |
| **C5** | **Counterfactual VLM rerank** for OOK named entities | VLM-as-judge with direct ranking is position-biased and brittle on long-tail entities | ~1 week | +5-10% R@1 on the long-tail slice | Backup |

The three primary contributions (`C1`, `C2`, `C4`) are designed to be **independent** - any one of them can ship without the others - and **additive** - the dev-set ablation will report each alone and stacked.

---

## 3. C1 - DiacriticBERT: a diacritic-robust late-interaction head

### 3.1 The problem this solves

Vietnamese ASR (PhoWhisper, Whisper-large-v3) and OCR (PaddleOCR PP-OCRv5, VietOCR) make systematic diacritic errors. A query like `"con cho` o+? cha?` Be^?n Tha`nh"` ("dog at Ben Thanh market") must match transcripts and OCR text that may contain `"con cho o cho Ben Thanh"` or `"con cho` o+? cha?` ben thanh"`. Off-the-shelf BGE-M3 was trained on clean Vietnamese; its similarity drops sharply when diacritics are missing or swapped. This is the failure mode explicitly called out as item 3 in `docs/strategy/00-master-strategy.md` SS 7 ("eight things that will lose this competition").

### 3.2 The method (novel)

1. **Construct a controlled diacritic-noise function.** For any clean Vietnamese string, generate K = 4 noisy variants by composing:
   - `drop_all_diacritics` (`tre? em` -> `tre em`)
   - `random_drop_p` (drop each diacritic mark with probability p sampled per-sentence from `Beta(2, 5)`)
   - `tone_swap_p` (swap one tone for another with probability p, sampled the same way)
   - `mixed` (drop + swap composed)
2. **Build a contrastive corpus.** Take ~500K clean Vietnamese captions / OCR text / ASR text from our index (post Qwen2.5-VL-7B captioning + cleaned PhoWhisper output). For each, produce 4 noisy variants. Yields ~2M (clean, noisy) positive pairs. Hard negatives = top-50 BGE-M3 neighbours of the clean side minus the positive set.
3. **Train a small late-interaction head on top of frozen BGE-M3.** Architecture: BGE-M3 frozen encoder + a 2-layer MLP projection (768 -> 384 -> 384) per side + ColBERT-style MaxSim scoring. LoRA r=8 on BGE-M3's last 4 layers is optional and ablated.
4. **Loss**: InfoNCE with temperature 0.05; in-batch negatives + 7 mined hard negatives per anchor.
5. **Training budget**: ~3 GPU-hours on one A6000 (250K steps batch 256).
6. **Inference plug-in**: at query time, the BGE-M3 sparse + dense scores are computed as today, and the DiacriticBERT MaxSim is added as a third score in the OCR/ASR/caption Elasticsearch fusion. Weight is learned by C2.

### 3.3 Why this is novel (not in prior art)

- LSC/VBS literature does not address diacritic noise as a first-class retrieval problem; it's mentioned as a *preprocessing* concern (PhoWhisper output post-processing with Gemini, in our own proposal 01 SS 5.6).
- BGE-M3 and PhoBERT-v2 are trained on clean Vietnamese; neither has been re-trained with a controlled diacritic-noise schedule that we are aware of. The closest published work is general typo-robust IR (Penha et al. 2022, Sidiropoulos & Kanoulas 2022) which targets English typos, not Vietnamese-specific tone/diacritic noise.
- We are not training a new backbone, only a small head. Risk is bounded.

### 3.4 Eval plan

- **Held-out diacritic-noise test set**: 500 queries with 4 noisy variants each (2K query instances). Metric: R@10 on the held-out clean target.
- **Real-task slice**: on the 300-task internal evaluation harness (proposal 05), tag the queries that involve proper nouns or OCR text. Report R@1 / R@5 with C1 on vs off.
- **Anti-overfitting guard**: per-class slicing - C1 must not regress *any* class (clean Vi caption retrieval included) by more than 1.5%.

### 3.5 Files

- `train/diacritic_bert.py`
- `train/diacritic_noise.py` (the noise function, fuzz-tested)
- `data/diacritic_pairs.parquet` (the contrastive corpus)
- `prompts/` - none; this is a pure embedding contribution.
- Eval lives in `eval/diacritic_robustness.py`.

### 3.6 If it doesn't work

If C1 does not show a >=1% R@1 lift on the OCR/ASR-bridged slice, ship the simpler fallback: pre-process all Vietnamese queries through SeaLLMs-v3 with prompt "Su+?a chi?nh ta? va` da^?u" before any retrieval. Cheap, no training, ~1-3% lift in our internal tests of similar pipelines. Document the negative result in `experiments/c1/RESULT.md`.

---

## 4. C2 - Learned per-task-type fusion (replacing RRF k=60)

### 4.1 The problem this solves

Proposal 01 SS 2.4 justifies RRF k=60 as "standard hyper; robust across distributions". RRF (Cormack 2009) is robust precisely because it ignores score magnitudes - which means it also ignores the *quality* of each ranker. We stack 12-15 ranked lists from very different distributions:

- 4 image-text dense ANN (SigLIP-2, Meta CLIP 2, InternVideo2, optional ColVintern) per Vietnamese paraphrase x 3 paraphrases = 12 lists
- BM25 + BGE-M3-dense + BGE-M3-sparse over OCR + ASR + caption = 9 lists
- Optional CLAP audio-event ranked list

Uniform RRF is *demonstrably* not the optimal weighting across all four task types (KIS, QA, TRAKE, Ad-hoc). KIS rewards image-text dense; QA leans on OCR/caption; Ad-hoc is the wild card.

### 4.2 The method (novel for this competition, not for IR research)

A per-task-type **learned linear fusion** trained by *coordinate descent on rank positions*:

1. For each task type t in {KIS, QA, TRAKE, Ad-hoc}, define weights `w_t in R^15` and a per-ranker score normalisation function (min-max, z-score, or rank-reciprocal).
2. The fused score of doc d in task t is `score(d, t) = sum_i w_t[i] * norm_i(rank_i(d))`.
3. Train on the 300-query internal dev set with the gold target. Use **LambdaRank objective** with NDCG@10 as the gain (lightgbm `LGBMRanker`, ~30 leaves, 200 trees, early-stop on a 20% val split). This is one notebook of work.
4. Per-task-type model is selected at runtime by the planner LLM's emitted intent JSON.
5. **Safety guardrail**: if learned fusion is more than 5% worse than RRF on a 50-query held-out slice, fall back to RRF k=60 automatically.

### 4.3 Why this is novel (in this competition)

- No LSC/VBS top-3 system in the 2022-2025 literature reports a learned fusion across this many heterogeneous rankers. Most use RRF or a hand-tuned linear blend per system release.
- The novelty isn't LambdaRank itself (Burges et al. 2007) - it's the *per-task-type* model selection driven by the planner LLM. The planner already emits `modality_priority` (proposal 01 SS 3) and `task_type`; we wire those into fusion model selection.

### 4.4 Eval plan

- **Per-task-type ablation**: report NDCG@10 for {RRF, single-global learned, per-task learned} on the 300-task dev set.
- **Robustness**: bootstrap CI of NDCG@10 with 1000 resamples; confirm per-task-type beats RRF by >=1 point with non-overlapping CI.
- **Train-time leakage check**: leave-one-task-out CV - train on 75% of tasks per type, test on the held-out 25%.

### 4.5 Files

- `train/learned_fusion.py` (LightGBM training + per-task-type model export)
- `src/retrieval/fusion.py` (runtime: load model, apply, fallback to RRF)
- `eval/fusion_ablation.py`

### 4.6 If it doesn't work

Worst case (overfit to dev set): we keep RRF and lose ~3 days. The runtime guardrail (auto-fallback if worse than RRF on a held-out slice) makes this risk-free in production.

---

## 5. C3 - PriorDP: story-graph generalisation of DANTE for TRAKE (backup)

### 5.1 The problem this solves

DANTE (AIO_Owlgorithms LSC 2025) is a 4-dimensional shortest-path with a linear penalty `lambda * temporal_distance_variance`. The penalty is uniform across scene transitions - it doesn't know that "kitchen -> dining room" is a much more common transition than "kitchen -> swimming pool" in the training corpus. There is a learnable prior here.

### 5.2 The method (novel)

1. From the offline pipeline, build a **scene-transition co-occurrence matrix** `P(scene_{i+1} | scene_i)` keyed on `place_label`, `adl_label`, and a coarse `object_tag` bucket. Smoothed with Laplace-1 prior.
2. The TRAKE DP cost becomes `cost(d_i, d_{i+1}) = -log(retrieval_score(d_{i+1})) - alpha * log(P(scene(d_{i+1}) | scene(d_i))) + lambda * |t_{i+1} - t_i - mu_gap|`, where `mu_gap` is the empirical median gap between consecutive scenes in the training corpus.
3. alpha and lambda are tuned on the dev set via grid search.

### 5.3 Why this is novel

DANTE's published formulation is purely penalty-on-time. Adding a learned scene-transition prior on top is a strict generalisation. To our knowledge no LSC team has published this.

### 5.4 Why it's backup, not primary

TRAKE is one of four task types and is **not confirmed** to remain in 2026 (proposal 01 SS 8 risk row). Spending 2 weeks on C3 is only worthwhile if Phase 2 has clear slack and TRAKE remains in scope.

### 5.5 Files

- `train/scene_transition_prior.py`
- `src/trake/prior_dp.py`
- `eval/trake_ablation.py`

---

## 6. C4 - Agent self-distillation from interactive operator traces

### 6.1 The problem this solves

The AIC2026 automatic track is genuinely new in 2026 (LSC ran "automatic" sub-tracks in 2024-25 but they were toys). Every team's planner LLM will start from zero examples on day one. SnapMind (MMM 2026) gives the architecture pattern, not the training data. The natural training signal is *our own operators' decisions during Phase 1 and Phase 2 mock-task practice*.

### 6.2 The method (novel for this domain)

1. **Instrumentation (already in proposal 05 SS 5):** every interactive-track session logs `(query, planner_intent_json, tool_calls_chosen, fusion_weights, rerank_decision, final_submission, gold_outcome)` to Parquet.
2. **Distillation corpus**: filter to sessions where the operator's final submission was *correct* AND time-to-submit was below the team median. Yields ~500-2000 high-quality (query, plan, outcome) triples after Phase 1-2.
3. **Training**: use **DSPy MIPRO** or **DSPy BootstrapFewShot** to optimise the planner prompt against this corpus, treating tool-choice F1 + downstream R@1 as the joint objective. DSPy is already a CHOSEN dependency (`07-approaches-catalog.md` M2).
4. **At inference**, the automatic-track planner uses the distilled prompt + a kNN retrieval of the 3 nearest past (query, plan) examples from the same corpus as in-context demonstrations. This is essentially a private few-shot library mined from our own play.
5. **Continual refresh**: after every Phase 3 preliminary round and every Phase 4 mock-finals, re-run DSPy with the new traces appended. This makes the agent literally learn from the operator week-over-week.

### 6.3 Why this is novel

- The automatic track did not exist as a serious sub-event before AIC2026. There is no prior LSC/VBS team that has published a planner trained on its own operator's traces. SnapMind's blueprint stops at the architecture.
- The closest published patterns are (a) Reflexion / self-improvement loops (Shinn et al. 2023), and (b) trajectory-distillation for LLM agents (e.g., Distilling Reasoning Capability into Smaller LLMs, 2024). Neither targets multimodal retrieval planners trained on the *same team's* interactive operator.
- Frame the contribution as: *"the interactive track's operator IS the labelling oracle for the automatic-track agent."*

### 6.4 Eval plan

- **Phase 2 baseline**: zero-shot SeaLLMs-v3-7B planner on the automatic-track mock-finals; record R@1, R@5, calibrated confidence, Brier.
- **Phase 2 + C4**: same eval after one DSPy optimisation round on the Phase 1 operator traces. Expected: >=10% R@1 lift on Ad-hoc + QA.
- **Phase 4 refresh**: re-eval after Phase 3 traces appended. Expected: monotone improvement.

### 6.5 Files

- `train/planner_distill.py` (DSPy harness)
- `data/operator_traces.parquet` (the rolling log)
- `src/agent/planner_prompt.py` (loads the distilled prompt + few-shots at startup)
- `eval/agent_automatic_ablation.py`

### 6.6 If it doesn't work

If the distilled planner regresses vs the zero-shot SeaLLMs-v3 prompt, we ship the zero-shot prompt and document the negative result. The instrumentation work (logging operator traces) is valuable regardless because it powers post-round debugging.

---

## 7. C5 - Counterfactual VLM rerank for OOK named entities (backup)

### 7.1 The problem this solves

Vintern-3B-beta and Gemini 2.5 Flash, used as direct rankers on a 3x3 grid (proposal 01 SS 5.9), are known to be **position-biased** (LLaVA-Interleave, Gemini eval literature 2024-25) and brittle on long-tail named entities ("the photo of little Nguyen Van A's dog at Ben Thanh market"). Direct "rank these 9 from best to worst" prompts produce wobbly answers on such cases.

### 7.2 The method (novel for this domain)

Replace direct ranking with **iterative counterfactual pruning**:

1. Take the top-9 candidates from C2-fused retrieval.
2. Prompt: *"Cho 9 anh sau, ha~y cho.n MOT anh KHONG phu hop nha^?t vo+'i ca^u truy va^?n: '<q>'. Tra? lo+`i so^? thu+' tu+. va` ly do nga^?n go.n."* ("Pick the ONE image that is LEAST consistent with the query.")
3. Eliminate that candidate; repeat on the remaining 8.
4. Stop when 3 candidates remain. These are the final top-3.
5. **Position-bias mitigation**: at each pruning step, shuffle the input order; run 3 votes at temperature 0.7; majority-vote the pruned image.

### 7.3 Why this is novel

- Counterfactual / "rank by elimination" prompting has been shown to outperform direct ranking on long-tail LLM-as-judge tasks (Liu et al. 2024 LLM-Eval, Wang et al. 2025 CoT-Critic). To our knowledge it has not been applied to multimodal retrieval rerank in the LSC/VBS context.
- The 3-vote majority + shuffle directly attacks the documented position-bias failure mode of VLM judges.

### 7.4 Why it's backup, not primary

Cost: each query now triggers 6 + 5 + 4 = 15 grid evals instead of 1, plus 3-vote ensembling = ~45 VLM calls. With Vintern-3B-beta on local GPU this is ~1.5s extra latency. Acceptable but not free. Only ship if dev-set ablation shows >=5% R@1 lift on the long-tail slice.

### 7.5 Files

- `src/rerank/counterfactual.py`
- `prompts/counterfactual_rerank.txt`
- `eval/rerank_position_bias.py`

---

## 8. Timeline integration

| Week | Phase | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|
| W1-3 | Phase 0 | - | - | - | - | - |
| W4-6 | Phase 1 | - | - | - | Instrument operator traces (no training yet) | - |
| W7 | Phase 2 | Build noise function + corpus | Build LightGBM harness | - | - | - |
| W8 | Phase 2 | Train DiacriticBERT v1 | Train per-task fusion v1 | (only if slack) build scene-transition matrix | - | - |
| W9 | Phase 2 | Ablate C1; ship if pass | Ablate C2; ship if pass | (only if slack) ablate C3 | DSPy round 1 on Phase 1 traces | - |
| W10 | Phase 2 | freeze | freeze | freeze or drop | Ablate C4; ship if pass | (if budget) build C5 |
| W11-12 | Phase 3 prelim | use in prelim | use in prelim | (if shipped) use | use in prelim | - |
| W13-14 | Phase 4 finals prep | - | retrain on prelim data | (if shipped) refresh prior | DSPy round 2 on prelim traces | (if shipped) tune |
| W15+ | Finals | - | - | - | - | - |

Owners (one engineer each so they run in parallel):
- C1: Vietnamese NLP Engineer
- C2: Lead Engineer / Retrieval Architect
- C3: Lead Engineer (if Phase 2 slack)
- C4: Operator-1 / ML Engineer (since they own the planner prompts anyway)
- C5: Operator-1 / ML Engineer (if budget)

## 9. Honest framing for the team and the press kit

The right narrative is:

> **Floor**: we have reproduced the 2026 LSC SOTA Vietnamese stack carefully (proposals 01-07). This puts us on the finalist line, which is necessary but not sufficient.
>
> **Edge**: on top of that floor we have shipped three original contributions:
> 1. **DiacriticBERT** - the first retrieval head explicitly trained on Vietnamese diacritic noise.
> 2. **Per-task-type learned fusion** - replacing the 2009-vintage RRF default that every other team uses.
> 3. **Agent self-distillation** - a pattern that only exists because AIC2026 is the first year with both an interactive and an automatic track.
>
> **Moat**: aggressive operator drills (>=20% of prep time) and a verification panel that prevents the 10-point penalty mistake.

Anything more than that is overclaim. Anything less is selling ourselves short.

## 10. Acceptance test for this proposal

By end of Phase 2 (mid-Aug), the team must be able to point at:
- A C1 ablation showing R@1 lift on the OCR/ASR-bridged dev slice (or a documented negative result + fallback shipped).
- A C2 ablation showing NDCG@10 lift over RRF on the 300-task dev set (or a documented negative result + RRF kept).
- A C4 ablation showing automatic-track R@1 lift after one DSPy round (or a documented negative result + zero-shot planner kept).

If at least two of three primary contributions pass their ablations, the strategy is differentiated. If only one passes, the strategy is partly differentiated and we lean harder on operator drills. If none pass, we are a competent but undifferentiated finalist - which is still a defensible position given the floor.

## 11. References

- DANTE original formulation: AIO_Owlgorithms LSC'25, see `docs/papers/lsc-systems/QUEST-DANTE_arxiv-2512.13169.pdf`.
- SnapMind blueprint: `docs/papers/agentic-retrieval/` and <https://doi.org/10.1007/978-981-95-6963-2_20>.
- LSC SOTA review (2022-2024): `docs/papers/lsc-systems/LSC-SOTA-review_arxiv-2506.06743.pdf`.
- DreamLIP synthetic-caption recipe: see proposal 04 SS 2.1 (reused as-is; the *Vietnamese caption corpus* generated is the artefact, not the recipe).
- DSPy MIPRO: <https://dspy-docs.vercel.app/>.
- LambdaRank (Burges et al. 2007); RRF (Cormack et al. 2009); ColBERT MaxSim (Khattab & Zaharia 2020).
- Position-bias in VLM judges: LLaVA-Interleave (Li et al. 2024), Gemini-eval bias literature 2024-25.
