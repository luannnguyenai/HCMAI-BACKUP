# Proposal 02 - Automatic-Track Agent

> The AIC2026 competition introduces a NEW automatic track where AI agents compete autonomously, without human-in-the-loop. This proposal describes how we build that agent on top of the substrate from proposal 01.
>
> **What is reused vs novel in this proposal:** the architecture (planner LLM + tool registry + critic loop + self-verification) reproduces the SnapMind (MMM 2026) blueprint. The novel additions are: (a) **agent self-distillation** from the interactive-track operator's traces via DSPy MIPRO (SS 14, contribution C4 - primary), (b) the **counterfactual VLM rerank** tool for OOK queries (SS 10.3, contribution C5 - backup), and (c) the **same C1/C2 retrieval substrate** as the interactive system. The full novelty proposal is [`08-original-contributions.md`](08-original-contributions.md).

## 1. Why this proposal exists

The competition rules state two evaluation modes:
- **Traditional (interactive)**: user drives a virtual assistant to process queries.
- **Automatic**: AI assistants compete autonomously.

Most teams will treat the automatic track as an afterthought. We will treat it as **the same engine driven by a different controller**. If we win the automatic track, we capture an entire prize that few teams will seriously compete for.

**What is reused from prior art and what we add on top:**
- The **architecture pattern** (planner -> tool registry -> critic loop -> self-verification) reproduces SnapMind (MMM 2026, <https://doi.org/10.1007/978-981-95-6963-2_20>). We do not pretend the pattern is ours.
- The **innovation** is what we *train the planner on*: the interactive-track operator's correct, fast submissions become a continually-refreshed distillation corpus via DSPy MIPRO. This pattern only exists because AIC2026 is the first year the interactive and automatic tracks coexist as serious sub-events. See SS 14 below and [`08-original-contributions.md`](08-original-contributions.md) SS 6.

## 2. Design principles

1. **Same indexes, same tools.** Reuse Milvus/Elasticsearch/embeddings from proposal 01. The only new code is the controller.
2. **Modular tool registry.** Each retrieval pathway is a named tool with a JSON-schema contract. The agent picks tools the way a human operator would pick menu items.
3. **Time budget aware.** Assume the automatic track has a hard 30-60s wall clock per query. Plan for parallel execution.
4. **Self-verifying.** After producing candidate, the agent re-examines with a different model (VLM-as-judge) and may retract.
5. **Token-economical.** Cheap planner for routine queries; escalate only when low-confidence.

## 3. Architecture

```
+-------------------------------------------------------------+
|  Query in (text / image / audio)                            |
|     |                                                       |
|     v                                                       |
|  +---------------------------------------------------+      |
|  |  Planner (SeaLLMs-v3-7B function-calling)         |      |
|  |  - parse intent into JSON                         |      |
|  |  - select tools                                   |      |
|  |  - emit DAG (parallel groups)                     |      |
|  |  - emit fusion weights                            |      |
|  |  - emit max retries                               |      |
|  +---------------------------------------------------+      |
|     |                                                       |
|     v                                                       |
|  +---------------------------------------------------+      |
|  |  Tool registry (LangGraph state machine)          |      |
|  |    text_retrieval       (SigLIP-2)                |      |
|  |    vi_text_retrieval    (Meta CLIP 2)             |      |
|  |    video_retrieval      (InternVideo2)            |      |
|  |    ocr_retrieval        (BGE-M3 + BM25)           |      |
|  |    asr_retrieval        (BGE-M3 + BM25)           |      |
|  |    caption_retrieval    (BGE-M3 + BM25)           |      |
|  |    object_filter        (YOLO/Places)             |      |
|  |    scene_filter         (Places365)               |      |
|  |    adl_filter           (LSC-ADL labels)          |      |
|  |    temporal_filter      (timestamp range)         |      |
|  |    image_query          (image embedding)         |      |
|  |    generative_visual_query  (SDXL -> image search)|      |
|  |    rrf_fuse             (rank fusion)             |      |
|  |    dante_temporal       (TRAKE DP)                |      |
|  +---------------------------------------------------+      |
|     |                                                       |
|     v                                                       |
|  +---------------------------------------------------+      |
|  |  Critic / VLM-as-judge (Vintern-3B-beta)          |      |
|  |   - rank top-10 with chain-of-thought             |      |
|  |   - emit confidence score                         |      |
|  +---------------------------------------------------+      |
|     |                                                       |
|     v                                                       |
|  +---------------------------------------------------+      |
|  |  Decision agent (loop)                            |      |
|  |   - if conf > 0.8: submit                         |      |
|  |   - elif retries < max: critique + re-plan       |      |
|  |   - else: submit best-so-far + flag low-conf      |      |
|  +---------------------------------------------------+      |
|     |                                                       |
|     v                                                       |
|  Submission to DRES                                         |
+-------------------------------------------------------------+
```

## 4. Tool registry contract

Every tool registered in LangGraph follows this Pydantic schema:

```python
class Tool(BaseModel):
    name: str
    description: str             # Vietnamese + English
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    cost_estimate_ms: int        # planner uses this for budgeting
    parallelisable: bool
    cacheable: bool

    def __call__(self, **inputs) -> dict: ...
```

The planner LLM is shown a JSON list of these tools at every step. It can chain them but cannot invent new ones.

Example: `text_retrieval`
```yaml
name: text_retrieval
description: "Tim khung hinh phu hop voi mo ta tieng Viet/Anh bang SigLIP-2"
inputs:
  query: str                  # Vietnamese or English text
  top_k: int = 200
  filters: dict | None
outputs:
  candidates: list[CandidateFrame]
cost_estimate_ms: 120
parallelisable: true
cacheable: true
```

## 5. Planner prompt template (pseudo)

```
You are a query planner for a Vietnamese multimedia retrieval system. Given a user query, decompose it into a DAG of tool calls. Output strict JSON conforming to the schema. Available tools: <tool_list>. Available filters: time, location, object, ADL. Budget: 30s wall clock.

Examples:
- KIS query "tre em chay nhay duoi mua o san choi truong hoc"
  -> DAG:
     [par]
       text_retrieval(query="children running in rain at school playground", top_k=200)
       vi_text_retrieval(query="tre em chay nhay duoi mua o san choi truong hoc", top_k=200)
       caption_retrieval(query="tre em mua", top_k=100)
     [seq]
       rrf_fuse(k=60)
       structured_filter(place_label in {school, playground})
       vlm_rerank(query="...", top_k=10)

User query: <q>
JSON plan:
```

## 6. Self-verification loop

After the first plan executes:
1. Top-1 candidate is passed to VLM-as-judge with a yes/no question: "La` ca^?u tra? lo+`i da?ng tin ca^.y cho ca^u ho?i '<q>' kho^ng?" + emit confidence.
2. If `confidence < 0.7`:
   - Critic LLM looks at the plan + top-10 + the rejected reason and emits a *patch* to the plan.
   - Re-execute the plan.
   - Up to 3 retries.
3. If `confidence >= 0.7` or `retries >= 3`: submit.

## 7. Confidence calibration

This is critical and often missed. We will:
1. Use ensemble of 3 VLM-judge runs at temperature 0.7; mean = score, std = uncertainty.
2. Map raw score -> calibrated probability via Platt scaling fit on a held-out 200-query set.
3. Set threshold at the 70th percentile of past confident-correct examples.

## 8. Cost budget per query

| Component | Cost / query | Budget |
|---|---|---|
| Planner (SeaLLMs-v3 local) | 0 (we own GPU) | unlimited |
| Tool execution (Milvus + ES) | 0 (we own infra) | <500 ms |
| Vintern-3B-beta rerank (local) | 0 (we own GPU) | <2 s |
| Gemini 2.5 Flash escalation | $0.50 in + $3.00 out per M tokens | $0.01 per query at avg 5K in/1K out |
| **End-to-end** | **<$0.02 / query**; <30 s | -- |

For ~50 queries in a finals round: <$1 in API costs. Trivially affordable.

## 9. Critical engineering tasks

1. **LangGraph state machine** wiring all tools + critic + retry loop.
2. **Tool description corpus** in Vietnamese (so the planner LLM uses Vi reasoning).
3. **Few-shot library** of high-quality plan examples; auto-mined from interactive-track operator logs.
4. **Confidence calibration**: held-out 200 queries with ground truth, Platt scaling.
5. **Replay infrastructure**: every plan + execution + result is logged to Parquet for nightly regression.

## 10. The "secret weapon" tools

Beyond the obvious retrieval tools, three unusual ones (10.1 and 10.2 are reused tricks; 10.3 is ours):

### 10.1 `generative_visual_query` (reused from NII-UIT VBS'25)
- NII-UIT's VBS'25-winning trick: when the planner detects a hard descriptive query, generate a synthetic image with **Stable Diffusion XL** (Vietnamese -> English translation first), then run **image-to-image similarity** against our keyframe pool.
- Especially powerful for OOK entities ("anh con cho` tha?ng be Nguye^~n Va(n A o+? cha? Be^?n Tha`nh" - "the picture of little Nguyen Van A's dog at Ben Thanh market") - SDXL can't draw Nguyen Van A's actual dog but can draw "a small dog with a child at a Vietnamese street market", which gets us in the right neighbourhood.

### 10.2 `oot_external_search` (reused, standard LSC fallback)
- For genuinely out-of-knowledge named entities, route to **Google Lens API** or **Bing Visual Search** with the query as a hint. Returns image URLs we can embed and match.
- Limited use (low quality + cost) but a non-zero recall boost on the toughest queries.

### 10.3 `counterfactual_vlm_rerank` (C5, ours - backup contribution)
- Direct "rank these 9 from best to worst" prompts to VLM-as-judge are documented to be position-biased and brittle on long-tail entities. We replace direct ranking with **iterative counterfactual pruning**: ask the VLM which candidate is *least* consistent with the query, eliminate it, repeat on the remaining 8, etc., with input-order shuffle and 3-vote majority at each pruning step.
- Cost: ~15 grid evals x 3 votes = ~45 VLM calls (~1.5s extra on local Vintern-3B-beta). Acceptable but not free.
- Ships only if dev-set ablation shows >=5% R@1 lift on the OOK / long-tail-entity slice.
- See [`08-original-contributions.md`](08-original-contributions.md) SS 7 for the full method.
- Files: `src/rerank/counterfactual.py`, `prompts/counterfactual_rerank.txt`.

## 11. How the agent differs from the interactive system

| Dimension | Interactive | Automatic |
|---|---|---|
| Human in the loop | yes | no |
| Time budget per query | 5 min (KIS) / 3 min (QA, Ad-hoc, TRAKE) | hard 30-60s |
| Submission penalty awareness | operator avoids | agent must self-verify |
| Iteration | many (operator clicks, refines) | up to 3 retries |
| Confidence threshold | implicit (operator gut) | explicit (Platt-calibrated) |
| Strange/figurative queries | operator improvises | planner falls back to ensemble |

## 12. Reading list for implementer
- `docs/research-notes/04-vietnamese-stack-and-agents.md` Part 2 (this proposal's parent)
- SnapMind (MMM 2026): <https://doi.org/10.1007/978-981-95-6963-2_20>
- Cascaded multimodal agent: `docs/papers/agentic-retrieval/CascadedMM-Agent_arxiv-2512.12935.pdf`
- Smart routing: `docs/papers/agentic-retrieval/SmartRouting_arxiv-2507.13374.pdf`
- LangGraph docs: <https://langchain-ai.github.io/langgraph/>

## 13. Acceptance test

The automatic agent should achieve **at least 70% of the interactive system's score on the same 30-query mock-finals set**. If it falls below 50%, it indicates the planner LLM is undertuned and needs DSPy optimisation (which is C4 below - so a failure here triggers C4 if it has not already shipped).

## 14. Agent self-distillation (C4, ours - primary contribution)

> Full method, eval, and risk discussion in [`08-original-contributions.md`](08-original-contributions.md) SS 6. This section explains how it wires into the automatic-track agent at runtime.

### 14.1 Pattern in one sentence

The interactive-track operator is the labelling oracle for the automatic-track planner.

### 14.2 Wiring

```
[ Phase 1-3 interactive sessions ]
         |
         v
[ Operator-trace logger (proposal 01 SS 5.12) ]
         |
         v
   data/operator_traces.parquet
         |
         v
[ Distillation corpus filter ]
   - keep sessions where final_submission was correct
   - keep sessions where time_to_submit < team-median
         |
         v
[ DSPy MIPRO optimiser ]
   - student: SeaLLMs-v3-7B planner prompt
   - teacher: distilled (query, plan) demonstrations
   - objective: tool-choice F1 + downstream R@1 (joint)
         |
         v
[ Distilled planner prompt + kNN few-shot index ]
         |
         v
   src/agent/planner_prompt.py  (loads both at startup)
```

### 14.3 Continual refresh schedule

| When | Triggered by | Corpus size (estimate) |
|---|---|---|
| End of Phase 2 (mid-Aug) | Week 9-10 of plan | ~500-800 traces |
| End of Phase 3 prelim (late Aug) | Prelim round logs | +200-400 traces |
| Mid Phase 4 (Sept 6) | Mock-finals drills | +200-400 traces |
| Eve of finals (Sept 11) | Final drill batch | +100-200 traces |

The DSPy refresh runs in <1 hour on a single A6000 (the planner is 7B, MIPRO does ~50 forward passes per candidate prompt, ~10 candidate prompts).

### 14.4 Safety guardrail

Every distilled prompt is A/B'd on a frozen 100-query held-out set before replacing the production prompt. If the new prompt does not beat the previous by >=2% R@1, the previous prompt is kept and the new corpus is added to a quarantine queue for human review.

### 14.5 Why this is novel and worth the slide

- The automatic track is new in 2026. SnapMind (MMM 2026) gives the architecture but not the training signal.
- The closest published patterns (Reflexion self-improvement, trajectory distillation for LLM agents) do not target multimodal retrieval planners trained on the same team's interactive operator.
- The narrative for the finals press kit / post-competition writeup: *"The interactive operator and the automatic agent share an engine and, by week 12, share a planner trained on the operator's own decisions. The two tracks compound each other rather than competing for engineering attention."*

### 14.6 Files

- `train/planner_distill.py` (DSPy harness)
- `data/operator_traces.parquet` (rolling log; schema in proposal 01 SS 5.12)
- `src/agent/planner_prompt.py` (loads distilled prompt + few-shot kNN at startup)
- `eval/agent_automatic_ablation.py` (per-task-type R@1, confidence calibration vs zero-shot baseline)
