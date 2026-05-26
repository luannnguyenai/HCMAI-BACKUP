---
id: ADR-0004
title: No Streamlit at finals; React + WebSocket operator console
status: Accepted
decided_on: 2026-05-25
deciders:
  - team lead
---

# ADR-0004 — No Streamlit at finals; React + WebSocket operator console

## Status

Accepted.

## Context

In the 2025 AIC HCMC competition our predecessor team built the operator console with **Streamlit + REST API**. The team reported in the May 25 meeting that the resulting UI took "a lot of time" per interaction and that this directly hurt the final score.

The root cause is structural, not a tuning issue: Streamlit re-runs the entire script on every interaction, each click is a full HTTP round-trip, and image bytes are re-serialised through the Python process on every rerender. Empirical estimate for last year's setup: **1.5–4 s per operator interaction**. At KIS scoring `100 ? 50·t/T ? 10·w` over a 300-second time limit, every 30 s of UI lag costs 5 points. Across a finals round with 6 KIS tasks and ~8 interactions each, that is 12–32 points lost to UI architecture before any model decision is made.

## Decision

The operator console is a **React 18 + Vite + TypeScript** SPA with:
- **Zustand** for state (no rerender cascade)
- **react-window / react-virtuoso** for the image grid (only visible thumbnails render)
- **WebSocket** transport (`ws://`) for bi-directional low-latency updates — never HTTP polling for hot paths
- **nginx** serving static JPEG thumbnails — image bytes never travel through Python
- **shadcn/ui + Tailwind** for components

Backend: **async FastAPI** with `asyncio.gather` for parallel tool execution; never sequential awaits.

LLM serving: **vLLM** or **SGLang** for local models; never `transformers.generate` in production.

We explicitly forbid Streamlit, Gradio, Reflex, and Jupyter notebooks as the operator UI for the finals path. They remain acceptable as dev-time tools.

**Hard latency SLO**: p50 end-to-end (keypress ? top-10 rendered) < 900 ms; p95 < 2 s. CI gates PRs that regress against this.

## Consequences

### Positive
- Per-interaction latency target drops from 1.5–4 s to <100 ms — directly recovers 12–32 points per round at unchanged model quality.
- Forces the team to staff a frontend role early (proposal 06 already names this owner).
- WebSocket transport gives us push-based UI updates from the backend — required for streaming partial results from the planner LLM.

### Negative
- Higher upfront engineering cost than Streamlit: ~1–2 weeks of frontend scaffolding vs ~1 day.
- Requires a frontend-capable team member.

### Neutral / observable
- The UI test plan in [`docs/proposals/06-ui-ux-design.md`](../proposals/06-ui-ux-design.md) §12 (Playwright e2e) becomes load-bearing.
- The latency SLO becomes a CI gate, not a soft target.

## Alternatives considered

- **NiceGUI** — Vue-based, WebSocket-native, easier than React — rejected because still server-side reactive, untested against our latency SLO, and team has more React experience.
- **Reflex** — Python-only fullstack — rejected: same server-side reactive model as Streamlit; no benchmarks proving it meets our SLO.
- **Optimise Streamlit with `st.session_state` + `st.cache_data` + `st.fragment`** — keep the Python-only stack — rejected: 30% improvement at best; the architecture is the bottleneck, not the tuning.
- **Gradio** — popular ML UI library — rejected: same fundamental rerun model as Streamlit; worse than NiceGUI for our use case.

## References

- [`docs/proposals/06-ui-ux-design.md`](../proposals/06-ui-ux-design.md)
- [`docs/strategy/00-master-strategy.md`](../strategy/00-master-strategy.md) §10 (UI stack resolved)
- LSC 2022–24 SOTA review — <https://arxiv.org/abs/2506.06743> §IV-D (operator timing analysis)
