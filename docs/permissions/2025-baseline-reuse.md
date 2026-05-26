# Permission record — reuse of the 2025 baseline (`ThanhToan2111/AIC_2026`)

> Filed under [ADR-0010 §3](../adr/ADR-0010-borrow-from-2025-baseline.md) (licence verification before merge). The original author of the baseline is a current member of the AIC2026 team, so the third bullet of that policy applies: explicit written permission from the author, recorded here.

## 1. Source

| Field | Value |
|---|---|
| Repo | <https://github.com/ThanhToan2111/AIC_2026.git> |
| Commit analysed | `c3c3545` (Initial clean commit, 2026-05-26) |
| Local mirror | `C:\Dev\AIC2026-SCOAI\AIC_2026` |
| Original author | ThanhToan — GitHub: [`ThanhToan2111`](https://github.com/ThanhToan2111) |
| Author status | AIC2026 team member (confirmed 2026-05-26) |
| Licence file in source | **None** — default GitHub repo, "all rights reserved" by default. This permission record is what authorises reuse. |

## 2. Scope of reuse

Items the team intends to take from this repository, with their target home in our spec-driven layout. Each item only lands on `main` after the relevant SPEC's PR ships, with the borrowing header described in [ADR-0010 §3 point 1](../adr/ADR-0010-borrow-from-2025-baseline.md).

| # | Source location | Target SPEC | Status |
|---|---|---|---|
| 1 | `streamlit_api.py:122-200` — DRES login + submit flow | [SPEC-0018](../specs/SPEC-0018-dres-integration.md) | Spec authored (Draft). Implementation pending. |
| 2 | `model/transnetv2_pytorch.py` + `model_weights/transnetv2-pytorch-weights.pth` (29 MB) — TransNetV2 wrapper and trained weight | SPEC-0003 (data ingestion) | Spec not yet authored |
| 3 | `api/api_server.py:43-134` — Pydantic request schemas (`KISSearchRequest`, `TemporalCombinateSearchRequest`, `EventSpec`, etc.) | SPEC-0008 / SPEC-0009 / SPEC-0011 | Specs not yet authored |
| 4 | `model/BLIP2_rerank.py` — BLIP-2 ITC/ITM reranker glue | SPEC-0010 (VLM reranker) — fallback path | Spec not yet authored |
| 5 | `streamlit_api.py:58-60` — bookmarks session-state primitive | SPEC-0012 (React operator console) — TRAKE staging | Spec not yet authored |
| 6 | `.gitignore` — explicit weight-allowlist policy | Adopted in our [`.gitignore`](../../.gitignore) | Already in place |

Items added later require a row appended here, signed off by the author.

## 3. Permission statement

> I, ThanhToan (GitHub `ThanhToan2111`), confirm that the items listed in §2 of this document may be reused in the `AIC2026-SCOAI` repository under attribution as required by [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md). I understand the borrowing-header convention and the `THIRD_PARTY.md` ledger requirement. I will be notified when a new row is appended to §2.

Signed-off-by: __________________________

Date: __________________________

(Until this is signed, no borrowed code lands on `main`. SPEC-0018 may remain in `Draft` and be developed against clean-room stubs.)

## 4. Asks of the author — bundled for the next 30-min team interview

Each ask below corresponds to an open question already filed in a research note or spec. Resolving them in one conversation unblocks several downstream items.

| # | Topic | Source open Q | Why it matters |
|---|---|---|---|
| 1 | Real 2025 finals score breakdown per task type (KIS / QA / TRAKE / Ad-hoc) | research-note 05 §7 Q3 | Calibrates which task types we should prioritise; informs SPEC-0011 priority. |
| 2 | Venue network reality — was DRES reachable from venue Wi-Fi? Was a wired LAN provided? Did anyone fall back to 4G/5G? | research-note 05 §7 Q4, ADR-0005 | Material for the LLM-path bakeoff's "primary network condition" choice and for [`infra/cloud/CHECKLIST.md`](../../infra/) (forthcoming). |
| 3 | DRES v2 endpoint details — stock open-source or AIC HCMC fork? Did `/api/v2/submit` accept the shape we expect for all four task types? | SPEC-0018 §9 Q1 | Unblocks SPEC-0018 implementation. |
| 4 | TRAKE submission body shape — was TRAKE actually submitted via the 2025 system? If yes, what was the JSON body? | SPEC-0018 §9 Q2 | Unblocks AC6 of SPEC-0018 (TRAKE submission). |
| 5 | Session-id transport — confirmed query parameter `?session=...`? Did header or cookie also work? | SPEC-0018 §9 Q3 | Implementation detail; affects retry/auth-refresh code. |
| 6 | DRES rate limiting per session on `/api/v2/submit` | SPEC-0018 §9 Q4 | Affects automatic-track agent retry policy ([proposal 02](../proposals/02-automatic-track-agent.md) §6). |
| 7 | DANTE-style tuning — what `decay_rate` value worked in practice in 2025? `max_gap_seconds`? `same_video_only` always True? | research-note 05 §7 Q5 | Seeds SPEC-0011 ? sweep with a sensible starting point. |
| 8 | What did you wish you had done differently? | open-ended | Anti-rework insurance for Phase 1. |
| 9 | Dataset access — can we get the 2025 ingestion outputs (TransNetV2 keyframes, ASR JSON, OCR text)? | research-note 05 §7 Q1 | Lets Phase 1 start ahead of June 25's official dataset release. |
| 10 | Latency numbers — measured per-interaction p50/p95 in 2025, even roughly? | research-note 05 §7 Q2 | Calibrates SPEC-0001 CI thresholds and gives the LLM-path bakeoff a real comparison anchor. |

Recommended format: 30-minute call, recorded. Notes filed under `docs/permissions/2025-baseline-interview-notes.md` after the call (created at that time, not now).

## 5. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-26 | team lead | Stub created. Pending signature in §3. Pending interview per §4. |
