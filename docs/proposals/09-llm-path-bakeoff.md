# Proposal 09 - LLM Path Bakeoff (Local 5070 vs Groq Cloud)

> Decided in the May 25 strategy meeting. Hardware reality changed: finals box is an **RTX 5070 (12 GB VRAM)**, not the originally-assumed A6000. Training/indexing reality changed in our favour: we have **GH200** access for offline work. This proposal locks down the methodology for choosing between local-only and cloud-augmented inference paths for the planner LLM.
>
> **Owner**: the team lead.
> **Deadline**: end of June 2026 (before Phase 2 fine-tuning starts).
> **Decision artifact**: `experiments/llm-path-bakeoff/REPORT.md`.

## 1. Why this exists

The May 25 meeting surfaced two facts that invalidate the original §5 hardware assumption in `docs/proposals/01-interactive-system-architecture.md`:

1. Live-demo box is an RTX 5070, 12 GB. No A6000.
2. Training/indexing has GH200 access — effectively unlimited offline compute.

These two changes don't break the strategy (proposal 08 contributions still ship), but they force a re-architecting of the **online inference path**. The meeting concluded:

> Use whatever GPU is available (currently GH200) for **offline** training/indexing, then bake off **local RTX 5070 inference** against **Groq cloud inference** for the online planner LLM and ship whichever path wins on a fair benchmark.

This proposal is that benchmark, with success criteria pre-registered before any numbers are seen.

## 2. The paths under test

Groq currently serves text LLMs only (Llama, Mistral, Qwen-text, DeepSeek — no VLM). The VLM-as-judge reranker therefore is **not** part of this bakeoff; it remains local Vintern-3B-beta on the 5070 unless a separate decision is made. The bakeoff covers four end-to-end paths:

| Path | Planner LLM | VLM Reranker | Local VRAM at finals |
|---|---|---|---|
| **A. All local** | `SeaLLMs-v3-7B AWQ-INT4` on the 5070 via **SGLang** (grammar-constrained JSON) | `Vintern-3B-beta INT4` local | ~9 GB |
| **B1. Hybrid Groq-70B** | **Groq Llama-3.3-70B-versatile** (OpenAI-compatible API) | `Vintern-3B-beta INT4` local | ~4 GB |
| **B2. Hybrid Groq-8B** | **Groq Llama-3.1-8B-instant** (cheaper, lower-quality) | `Vintern-3B-beta INT4` local | ~4 GB |
| **C. All cloud** | **Groq Llama-3.3-70B-versatile** | **Gemini 2.5 Flash** | ~2 GB (text encoders only) |

A and B variants are the realistic candidates. C is included as the "high-quality upper bound" for plan quality reference.

## 3. Pre-registered success criterion (locked before seeing numbers)

```
SELECTED = path that satisfies ALL of the following:

  1. p95 end-to-end latency (keypress -> top-10 visible) < 2.0 s
  2. failure rate < 0.5 % over the 300-task benchmark
  3. valid_json_rate >= 99.0 %  (planner emits parseable JSON)
  4. retrieval R@10 within 1 % of the best path
  5. cost per finals round (50 queries) < $1.00

Among paths that satisfy 1-5, the WINNER is the path with the LOWEST p95
end-to-end latency. Ties broken by R@10, then by failure rate, then by
ALL-LOCAL (deterministic) preference.
```

If multiple paths pass, we ship the lowest-p95. If exactly one path passes, that's the answer. If zero pass, we fall back to **path A all-local** as the deterministic floor and escalate to the team to discuss what to do about R@10 / valid_json gaps.

## 4. Benchmark workload

The 300-query mock-task set from `docs/proposals/05-evaluation-harness.md` §3:

- 100 KIS (each with a Vietnamese description + ground-truth frame)
- 100 QA (each with a Vietnamese question + expected answer + supporting frames)
- 100 Ad-hoc (each with a Vietnamese category + curated relevant-frame set)
- TRAKE excluded from this bakeoff because it dominates wall-clock with DANTE DP; latency comparison would be confounded.

Same queries, same retrieval substrate (Milvus + Elasticsearch indexes), same VLM reranker step. The only thing that varies across paths is the **planner LLM** (and in path C, the VLM reranker).

## 5. Network conditions to test

| Profile | Description | Why |
|---|---|---|
| **N0. LAN local** | Same machine; not relevant for path A but ground-truth zero-network for cloud paths | Sanity baseline |
| **N1. 5G reliable ISP** (Vietnam) | Owner's home 5G — **the primary test condition** | Closest to a well-prepared finals deployment |
| **N2. Office fibre** (if available) | Backup for variance check | Controls for 5G-specific quirks |
| **N3. Mobile 4G hotspot** | Phone tether | Realistic worst-case fallback at venue |
| **N4. Throttled** (`tc netem`: 5 Mbps + 300 ms latency + 1 % packet loss) | Synthetic venue-Wi-Fi worst-case | Catches degradation we can't see in N1-N3 |

All four cloud-touching paths (B1, B2, C) are tested against N1, N3, N4. Path A tested only on N0 (no network involved).

Each (path, network) cell is run **3 times at different times of day** (morning Vietnam time = US night, afternoon = US morning, evening = US afternoon) to capture Groq peak-load latency variance.

## 6. Metrics captured per query

```
- planner_ttft_ms       Time to first planner-LLM token
- planner_total_ms      Time until full JSON output is parseable
- valid_json            Boolean: did the planner emit valid JSON?
- json_repair_rounds    How many retries did we need to get valid JSON?
- tool_exec_ms          Time for parallel tool execution (this should be ~constant across paths)
- rerank_ms             VLM reranker latency
- end_to_end_ms         keypress -> top-10 rendered in the UI
- ok                    Boolean: did the query complete without exception?
- failure_kind          timeout | rate_limit | 5xx | json_parse_error | other | (none)
- top10_correct         Boolean: did the ground-truth frame appear in our top-10?
- r_at_10               Position-aware: 1 if rank 1, 0.5 if rank 2-5, 0.1 if rank 6-10, 0 otherwise
- api_cost_usd          Sum of input+output token cost for this query
```

Aggregated over the 300 queries per (path, network, time-of-day) cell:

```
- p50, p95, p99 of {planner_ttft_ms, planner_total_ms, end_to_end_ms}
- mean valid_json_rate
- mean failure rate by failure_kind
- mean R@10 across queries
- mean api_cost_usd; extrapolated cost per 50-query round
```

## 7. Implementation skeleton

```python
# bin/benchmark_llm_paths.py
import asyncio, time
from pathlib import Path

PATHS = {
    "A_local":      LocalSGLangPlanner(
                        model="SeaLLMs/SeaLLMs-v3-7B-Chat",
                        quantization="awq_int4",
                        grammar="strict_json_schema",
                    ),
    "B1_groq_70b":  GroqPlanner(model="llama-3.3-70b-versatile"),
    "B2_groq_8b":   GroqPlanner(model="llama-3.1-8b-instant"),
    "C_all_cloud":  GroqPlanner(model="llama-3.3-70b-versatile"),  # VLM swap is in execute_plan()
}

NETWORK_PROFILES = ["N1_5g", "N3_4g_hotspot", "N4_throttled"]
TIMES_OF_DAY = ["morning_vn", "afternoon_vn", "evening_vn"]
tasks = load_eval_tasks("eval/mock_tasks_300.jsonl")

for path_name, planner in PATHS.items():
    for net in NETWORK_PROFILES if planner.uses_network else ["N0_local"]:
        for tod in TIMES_OF_DAY if planner.uses_network else ["any"]:
            metrics = []
            for task in tasks:
                t0 = time.perf_counter()
                try:
                    plan = await asyncio.wait_for(planner.plan(task.query), timeout=10.0)
                    valid_json = bool(plan)
                    top10 = await execute_plan(plan, vlm=planner.preferred_vlm())
                    r10 = score_r_at_10(top10, task.ground_truth)
                    ok, failure = True, None
                except asyncio.TimeoutError:
                    ok, failure, valid_json, r10 = False, "timeout", False, 0.0
                except RateLimitError:
                    ok, failure, valid_json, r10 = False, "rate_limit", False, 0.0
                except json.JSONDecodeError:
                    ok, failure, valid_json, r10 = False, "json_parse_error", False, 0.0
                except Exception:
                    ok, failure, valid_json, r10 = False, "other", False, 0.0
                metrics.append({
                    "task_id": task.id, "ok": ok, "failure": failure,
                    "ttft_ms": planner.last_ttft_ms,
                    "planner_total_ms": planner.last_total_ms,
                    "end_to_end_ms": (time.perf_counter() - t0) * 1000,
                    "valid_json": valid_json,
                    "r_at_10": r10,
                    "api_cost_usd": planner.last_call_cost_usd,
                })
            persist(path_name, net, tod, metrics)

render_report("experiments/llm-path-bakeoff/REPORT.md",
              results, decision_criterion=PRE_REGISTERED_CRITERION)
```

Owner builds this on top of `bin/eval` from proposal 05. The harness already loads the 300-task set and runs the retrieval substrate; only the planner-swap layer is new.

## 8. The decision matrix template (filled in after the runs)

```
              p50 EtoE  p95 EtoE  fail%   valid_json%  R@10   $/round  PASS?
A_local       _____ms   _____ms   __%    ___%          ___    $0.00     ?
B1_groq_70b   _____ms   _____ms   __%    ___%          ___    $____     ?
B2_groq_8b    _____ms   _____ms   __%    ___%          ___    $____     ?
C_all_cloud   _____ms   _____ms   __%    ___%          ___    $____     ?

PRE-REGISTERED CRITERIA (each path must pass ALL):
  - p95 EtoE   < 2000 ms
  - fail%      < 0.5 %
  - valid_json >= 99.0 %
  - R@10 within 1.0 % of the best path
  - $/round    < $1.00

WINNER = lowest p95 among PASS paths.
TIE-BREAKER ORDER: R@10 > fail% > all-local preference.
```

## 9. Risks the bakeoff must NOT hide

1. **Groq peak-load tax.** Groq's free-tier latency degrades sharply during US business hours. The 3-times-of-day matrix catches this; the report **must** show p95 *per time-of-day*, not just aggregated.
2. **Plan-quality drift.** Llama-3.3-70B (Groq) and SeaLLMs-v3-7B (local) will produce *different* tool DAGs for the same Vietnamese query. Latency is meaningless if R@10 collapses. R@10 is **gated, not optional**.
3. **JSON-strictness gap.** SGLang's grammar-constrained decoding guarantees valid JSON. Groq has no equivalent; we'll handle malformed JSON with a retry-and-repair shim. **valid_json_rate** must be measured *after* the repair shim — if repair is needed even 1 % of the time it erodes the latency win.
4. **Cold start.** Path A must benchmark with the vLLM/SGLang server pre-warmed (`POST /v1/completions` with a dummy query) — this is what we'll do at finals. Don't measure cold latency unless you also measure cloud cold-DNS-resolution latency.
5. **Token-count sensitivity.** Cloud cost scales with tokens. The benchmark must record realistic input sizes — full Vietnamese paraphrases + tool registry description in the prompt, not toy inputs.

## 10. Network condition: 5G owner-supplied baseline

The bakeoff owner has confirmed access to **5G from a reliable Vietnamese ISP** as the primary cloud-path test environment. This is N1, the headline number in the report. N3 (mobile 4G) and N4 (throttled) are stress tests; we don't expect to pass on them and we don't need to — we need to pass on N1 because *that's the network we plan to bring*.

**Implication for finals deployment**: if a cloud path wins, the team brings the same 5G-equipped device + a known-good ISP SIM card to the venue, with the laptop's onboard radio as primary and a phone hotspot as fallback. Wi-Fi at the venue is never trusted.

## 11. Output of the bakeoff

By **end of June 2026**:

1. `experiments/llm-path-bakeoff/REPORT.md` - the filled decision matrix + per-condition analysis + winner declaration.
2. `experiments/llm-path-bakeoff/raw/*.parquet` - per-query metrics so we can re-analyze.
3. A 5-bullet summary added to `docs/strategy/00-master-strategy.md` §10 (open questions) updating the cloud-budget question with the actual decision.
4. If the cloud path wins, a `infra/cloud/` checklist for venue deployment (DNS pre-warm, SIM card, 5G test on arrival).
5. If the local path wins, the AWQ-INT4 quantized SeaLLMs-v3-7B weights staged on the 5070 finals box, with quantization-quality regression check on the dev set.

## 12. Owner responsibilities

The owner of this proposal (the team lead) commits to:

- [ ] Stand up both serving stacks (vLLM/SGLang local + Groq client) by June 10.
- [ ] Run the full matrix (4 paths × 3 networks × 3 times-of-day = 36 cells) by June 20.
- [ ] Write `experiments/llm-path-bakeoff/REPORT.md` by June 25.
- [ ] Present the decision to the team in the Phase 1 → Phase 2 transition meeting (last week of June).
- [ ] Update proposal 01 SS 5.8 (Planner LLM) and proposal 02 SS 4 (Tool registry contract) with the chosen path.
- [ ] Pre-register the success criterion in this file as a frozen artifact (this section already does so); the writeup must reference this commit by SHA.

## 13. Anti-bias safeguards

The owner is a stakeholder in the outcome and the only person running the benchmark. To prevent post-hoc rationalization:

- The decision matrix in §8 and the criteria in §3 are **frozen** at the SHA that commits this proposal. They cannot be changed after the first benchmark run starts.
- The raw per-query metrics are committed to git (`experiments/llm-path-bakeoff/raw/`). Any teammate can re-aggregate them and challenge the report.
- The team-internal Phase 1 → Phase 2 meeting is the formal review of the decision. Disagreement triggers a re-run of specific (path, network, time-of-day) cells, not a re-spec of the criteria.

## 14. References

- `docs/proposals/01-interactive-system-architecture.md` §5.8 - planner LLM as currently specified.
- `docs/proposals/02-automatic-track-agent.md` §4 - tool registry contract that the planner emits against.
- `docs/proposals/05-evaluation-harness.md` §3 - the 300-task mock set this bakeoff reuses.
- `docs/proposals/08-original-contributions.md` §4 - C2 learned fusion is downstream of the planner; chosen path must support C2's runtime fallback to RRF.
- `docs/strategy/00-master-strategy.md` §10 - cloud budget question; this proposal closes it.
- Groq API docs - <https://console.groq.com/docs>
- SGLang grammar-constrained decoding - <https://docs.sglang.ai/sampling_params.html>
