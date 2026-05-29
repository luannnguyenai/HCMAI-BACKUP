# ADR Registry

> Append-only record of architectural decisions. Never reuse an ID; never edit a Accepted ADR's Decision/Context — supersede it instead. See [`CONTRIBUTING.md`](../../CONTRIBUTING.md) for the workflow.

## Conventions

- ID format: `ADR-NNNN` (4 digits).
- File path: `docs/adr/ADR-NNNN-kebab-name.md`.
- Status one of: `Proposed` | `Accepted` | `Superseded by ADR-NNNN`.

## ADRs

| ID | Title | Status | Decided | Notes |
|---|---|---|---|---|
| [ADR-0001](ADR-0001-floor-edge-moat-strategy.md) | Strategy framed as Floor / Edge / Moat | Accepted | 2026-05-24 | Reframes the four-advantage stack |
| [ADR-0002](ADR-0002-vietnamese-capable-not-native.md) | "Vietnamese-capable" not "Vietnamese-native" | Accepted | 2026-05-24 | Honest framing of model stack |
| [ADR-0003](ADR-0003-rtx5070-finals-gh200-offline.md) | RTX 5070 at finals, GH200 for offline | Accepted | 2026-05-25 | Forces INT4 quantization on hot path |
| [ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md) | No Streamlit; React + WebSocket UI | Accepted | 2026-05-25 | Closes last year's latency root cause |
| [ADR-0005](ADR-0005-llm-path-bakeoff-gates-planner.md) | LLM path chosen by bakeoff, not opinion | Accepted | 2026-05-25 | Closes "cloud budget" open question |
| [ADR-0006](ADR-0006-int4-quantization-hot-path.md) | INT4 / FP4 quantization for hot-path models | Accepted | 2026-05-25 | Required by ADR-0003 |
| [ADR-0007](ADR-0007-original-contributions-c1-c2-c4.md) | Three primary original contributions: C1, C2, C4 | Accepted | 2026-05-24 | The "Edge" tier of ADR-0001 |
| [ADR-0008](ADR-0008-rrf-as-runtime-fallback.md) | RRF is the fallback; C2 is the default fusion | Accepted | 2026-05-24 | Inverts the original proposal stance |
| [ADR-0009](ADR-0009-sdd-workflow.md) | Spec-Driven Development is the workflow | Accepted | 2026-05-26 | The workflow itself |
| [ADR-0010](ADR-0010-borrow-from-2025-baseline.md) | Borrow from the 2025 baseline repo under attribution | Accepted | 2026-05-26 | Policy + per-file header + `THIRD_PARTY.md` |
| [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md) | Cloudflare R2 as the artifact store; 4-tier persistence across lease rollovers | Accepted | 2026-05-29 | Implemented in [SPEC-0022](../specs/SPEC-0022-remote-gpu-runner.md) |
