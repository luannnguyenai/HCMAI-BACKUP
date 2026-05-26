---
id: SPEC-0002
title: LLM path bakeoff runner (bin/benchmark_llm_paths)
status: Draft
owner: team lead
created: 2026-05-26
updated: 2026-05-26
implements_proposal: docs/proposals/09-llm-path-bakeoff.md
related_adrs:
  - ADR-0003
  - ADR-0005
  - ADR-0006
depends_on:
  - SPEC-0001
---

# SPEC-0002 — LLM path bakeoff runner

> The benchmark script that decides whether the planner LLM runs locally on the RTX 5070 (SGLang + SeaLLMs-v3-7B INT4) or via Groq cloud. Implements [`docs/proposals/09-llm-path-bakeoff.md`](../proposals/09-llm-path-bakeoff.md). Owned by the team lead; deadline end of June 2026.

## 1. Context

[`docs/proposals/09-llm-path-bakeoff.md`](../proposals/09-llm-path-bakeoff.md) defines the bakeoff methodology and pre-registers the success criterion. This spec is the buildable runner that produces the decision artifact `experiments/llm-path-bakeoff/REPORT.md`.

ADR-0005 commits us to choosing by measurement, not opinion. ADR-0003 fixes the inference target as RTX 5070; ADR-0006 fixes the quantization as INT4/FP4.

## 2. Scope

### 2.1 In scope
- A CLI `bin/benchmark_llm_paths` that runs the full 4-path × 3-network × 3-time-of-day matrix and emits `experiments/llm-path-bakeoff/REPORT.md`.
- An OpenAI-compatible client abstraction so each path is configured via JSON, not code.
- Integration with SPEC-0001's eval harness for the per-query R@10 measurement.
- Raw per-query metrics persisted to git-trackable Parquet.
- Decision-matrix rendering against the criteria pre-registered in proposal 09 §3.

### 2.2 Out of scope
- Standing up the underlying serving stacks (SGLang + SeaLLMs-v3-7B, Groq client) — those are SPEC-0008's job; this spec consumes them.
- The retrieval substrate (Milvus + Elasticsearch + indexes) — those are SPEC-0003 / SPEC-0006 / SPEC-0007; this spec consumes them too.
- Acting on the decision (updating downstream specs, deploying weights) — those are post-decision tasks tracked separately.

## 3. API contract

### 3.1 CLI

```
bin/benchmark_llm_paths [OPTIONS]

Options:
  --config PATH          Path to bakeoff config YAML (default: configs/bakeoff.yaml).
  --paths PATH_LIST      Comma-separated path IDs to run (default: all in config).
                         e.g. "A_local,B1_groq_70b"
  --networks NET_LIST    Comma-separated network profile IDs (default: all in config).
  --times TIME_LIST      Comma-separated time-of-day windows (default: all in config).
  --tasks PATH           Mock-task set; default tests/mock_tasks/full_300.jsonl.
  --output DIR           Output directory; default experiments/llm-path-bakeoff/raw/
  --report-only          Skip benchmarking; just regenerate REPORT.md from existing raw data.
  --pin-criteria-sha SHA Override the criteria-frozen-at-SHA. Default: read from
                         git for `docs/proposals/09-llm-path-bakeoff.md`.
```

### 3.2 Config schema (`configs/bakeoff.yaml`)

```yaml
criteria_frozen_at_sha: "<commit SHA of proposal 09>"

paths:
  - id: A_local
    description: "vLLM/SGLang + SeaLLMs-v3-7B AWQ-INT4 (local 5070)"
    planner:
      type: openai_compatible
      base_url: http://localhost:30000/v1
      model: SeaLLMs/SeaLLMs-v3-7B-Chat
      api_key: sk-noauth
      extra_params:
        grammar: strict_json_schema
    reranker_vlm:
      type: local_vllm
      base_url: http://localhost:30001/v1
      model: 5CD-AI/Vintern-3B-beta

  - id: B1_groq_70b
    description: "Groq Llama-3.3-70B (cloud) + Vintern-3B local"
    planner:
      type: openai_compatible
      base_url: https://api.groq.com/openai/v1
      model: llama-3.3-70b-versatile
      api_key: ${env:GROQ_API_KEY}
    reranker_vlm:
      type: local_vllm
      base_url: http://localhost:30001/v1
      model: 5CD-AI/Vintern-3B-beta

  # ... B2_groq_8b, C_all_cloud configured similarly

networks:
  - id: N1_5g
    description: "Vietnamese 5G, owner-provided"
    netem: null
  - id: N3_4g_hotspot
    description: "Phone 4G tether"
    netem: null
  - id: N4_throttled
    description: "tc netem: 5 Mbps + 300 ms latency + 1% loss"
    netem:
      rate: 5mbit
      delay_ms: 300
      loss_pct: 1.0

time_of_day:
  - { id: morning_vn,   hours_local: [6, 11] }
  - { id: afternoon_vn, hours_local: [12, 17] }
  - { id: evening_vn,   hours_local: [18, 22] }

success_criteria:
  p95_end_to_end_ms_max: 2000
  failure_rate_max: 0.005
  valid_json_rate_min: 0.99
  r_at_10_within_pct_of_best: 1.0
  cost_per_50_query_round_max_usd: 1.00
```

### 3.3 Per-query metric record (Parquet schema)

```python
class BakeoffQueryMetrics(BaseModel):
    bakeoff_run_id: str
    path_id: str
    network_id: str
    time_of_day_id: str
    task_id: str
    task_type: str

    planner_ttft_ms: float | None
    planner_total_ms: float | None
    valid_json: bool
    json_repair_rounds: int

    tool_exec_ms: float
    rerank_ms: float
    end_to_end_ms: float

    ok: bool
    failure_kind: Literal["timeout","rate_limit","5xx","json_parse_error","other"] | None

    top_10_correct: bool
    r_at_10: float

    api_cost_usd: float
    timestamp_utc: str          # ISO 8601
```

## 4. Behaviour

### 4.1 Run flow
1. Validate config; verify the `criteria_frozen_at_sha` matches the actual current commit SHA of proposal 09 (refuse to run if drift detected, unless `--pin-criteria-sha` overrides).
2. For each (path, network, time-of-day) cell:
   - If network profile has `netem` config, apply it via `tc qdisc add` (sudo). De-apply at cell-end.
   - Wait until the local clock is inside `hours_local` for the time-of-day window. If `--no-wait-tod` is set, skip the wait and tag the cell as `tod_skipped`.
   - Boot the path's planner client + VLM-reranker client. Smoke-test with one dummy query.
   - For each task in `--tasks`: run the full pipeline (planner ? tool exec ? rerank ? top-10), capture all metrics, persist one Parquet row.
   - Tear down the path.
3. After all cells: aggregate, render `REPORT.md`, write `summary.json`.

### 4.2 Failure semantics
- Per-query failures (timeout, rate limit, 5xx, malformed JSON after `max_repair_rounds` retries) are *recorded*, not aborted. The run continues.
- Per-cell setup failure (e.g. SGLang server won't boot) aborts that cell and tags it `setup_failed` in the report.
- A cell that produces fewer than 95 % of expected per-task rows is flagged as `incomplete` and excluded from the pass/fail decision matrix.

### 4.3 Decision rendering
`REPORT.md` includes:
- The full decision matrix from proposal 09 §8, with PASS/FAIL flag per criterion per path.
- Per-(network, time-of-day) sub-tables for paths that touch the network.
- The frozen-at-SHA commit reference and a git-blame link.
- A "winner declared" section: lowest p95 among paths that satisfy all 5 criteria. If zero paths pass, declare "no winner — defer to all-local fallback per ADR-0005."

## 5. Acceptance criteria

- **AC1**: `bin/benchmark_llm_paths --config configs/bakeoff.yaml --paths A_local --networks N1_5g --times morning_vn --tasks tests/mock_tasks/smoke_20.jsonl` runs to completion against a fake-planner stub and produces a `summary.json` + `REPORT.md`.
- **AC2**: A `criteria_frozen_at_sha` that does not match the current proposal-09 SHA aborts with exit code 3, unless `--pin-criteria-sha` is supplied. The abort message names both SHAs.
- **AC3**: Per-query Parquet rows conform to the schema in §3.3 and are appended to `experiments/llm-path-bakeoff/raw/run-<run_id>.parquet`.
- **AC4**: A path with planner failure rate ? 0.5 % is marked `FAIL — failure_rate` in the decision matrix.
- **AC5**: A path with `valid_json_rate < 99 %` is marked `FAIL — valid_json` in the decision matrix.
- **AC6**: A path with `R@10 > best_r_at_10 + 1.0 %` (worse by more than 1.0 percentage points) is marked `FAIL — quality_floor` in the decision matrix.
- **AC7**: A path whose p95 end-to-end exceeds 2 s in at least one (network, time-of-day) cell is marked `FAIL — latency` for that cell. The path passes overall only if at least 8 of 9 cells (network × tod) pass.
- **AC8**: `--report-only` regenerates `REPORT.md` from existing Parquet rows without launching any LLM calls or network probes.
- **AC9**: When zero paths pass all five criteria, the report's "winner declared" section reads "no winner — defer to all-local fallback per ADR-0005" verbatim.
- **AC10**: The `summary.json` includes a `frozen_criteria_sha` field equal to the SHA the run was validated against.

## 6. Non-functional requirements

- **Wall clock**: full matrix (4 paths × 3 networks × 3 tod × 300 tasks = 10,800 runs) completes within **3 calendar days**. Note: this is *calendar*, not GPU-hours — the time-of-day matrix forces real-world spacing.
- **GPU**: A_local cells require the RTX 5070 (or 4090 / A6000 as substitute during dev) with SGLang + Vintern running.
- **Network**: A_local cells need none. B1/B2/C cells need real network access; netem is applied via `tc` (requires root).
- **Cost ceiling**: total cloud spend across the full matrix < $50.
- **Data**: raw Parquet output ? 100 MB total; committable to git.

## 7. Dependencies

- **Internal**: SPEC-0001 (eval harness — provides the task loader, the retrieval substrate, the metric definitions).
- **External**:
  - `openai >= 1.40` (OpenAI-compatible client) — works against Groq, vLLM, SGLang.
  - `tc` / `iproute2` (Linux) for netem application.
  - `groq` SDK optional; if installed, used for native error types.
  - SGLang ? 0.5 with grammar-constrained decoding.
  - vLLM ? 0.7 with Vintern-3B support (or HF transformers fallback).
- **Data**:
  - `tests/mock_tasks/full_300.jsonl` (from SPEC-0001).
  - `configs/bakeoff.yaml` checked in.

## 8. Test plan

### 8.1 Unit tests (`tests/unit/test_bakeoff_runner.py`)
- `test_criteria_sha_mismatch_aborts_AC2`
- `test_parquet_schema_conforms_AC3`
- `test_decision_matrix_failure_rate_FAIL_AC4`
- `test_decision_matrix_valid_json_FAIL_AC5`
- `test_decision_matrix_quality_floor_FAIL_AC6`
- `test_decision_matrix_latency_per_cell_AC7`
- `test_report_only_uses_existing_parquet_AC8`
- `test_no_winner_message_verbatim_AC9`

### 8.2 Integration tests (`tests/integration/test_bakeoff_end_to_end.py`)
- Spin up a fake-planner stub that emits canned latencies and JSON; run the smoke 20-task set against it; assert REPORT renders with deterministic pass/fail.

### 8.3 Manual / live
- After the smoke flow passes in CI, the team lead runs the full matrix per the deadline in ADR-0005.

## 9. Open questions

- **Q1**: Time-of-day enforcement at dev time. The 3-tod matrix takes ~3 calendar days end-to-end. Should `--no-wait-tod` flag be allowed during dev (tagged in the report) or banned? Recommend: allowed in dev, banned in CI via env-var check.
- **Q2**: Network condition N4 (`tc netem`) requires sudo. Should the harness shell out, or should we document a separate "operator-runs-tc-manually" pre-step? Recommend: shell out, but with an explicit `--require-sudo` confirmation flag.
- **Q3**: Do we add a fifth path (e.g. Together AI + Llama) post-spec if interesting numbers emerge from A/B/C? Recommend: yes via a new SPEC-NNNN that extends this; do not retro-edit the bakeoff config.
- **Q4**: Where does `frozen_criteria_sha` get displayed in the operator UI later? Recommend: in `infra/cloud/CHECKLIST.md` for venue deployment, generated post-decision.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-26 | team lead | Created (Draft) |
