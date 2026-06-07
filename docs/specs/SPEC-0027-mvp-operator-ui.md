---
id: SPEC-0027
title: MVP operator UI (React + WebSocket KIS console)
status: Approved
owner: unassigned
created: 2026-06-05
updated: 2026-06-07
implements_proposal: docs/proposals/06-ui-ux-design.md
related_adrs:
  - ADR-0004
  - ADR-0013
  - ADR-0015
depends_on:
  - SPEC-0026
---

# SPEC-0027 - MVP operator UI

> The minimal React + WebSocket KIS console served by the shared MVP server
> (ADR-0013): a Vietnamese query box, a virtualised ranked keyframe grid, a
> frame-detail view, a lane selector, and an in-UI issue-capture control. It is
> the MVP subset of the full operator console in proposal 06; TRAKE, QA, the
> planner panel, the submission-verification bar, and novice mode are out of
> scope. Its only backend is the SPEC-0026 API.

## 1. Context

[ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md) mandates a
React 18 + Vite + TypeScript SPA with WebSocket transport, Zustand state, a
virtualised image grid, and static (nginx) image serving - no Streamlit.
Proposal 06 is the full operator-console design. This spec carves out the
smallest slice that lets the team try Vietnamese KIS retrieval against the
SPEC-0026 API on the AIC2025 proxy corpus (121,457 keyframes, no ground-truth
answer key, so correctness is judged qualitatively by the tester). The
issue-capture control exists so testers file reproducible reports (query +
lanes + returned frames + screenshot + timestamp) instead of describing bugs
from memory.

There is no frontend in the repo today (no `package.json`, no `web/` tree); this
spec introduces the frontend toolchain as a net-new addition under `web/`.

## 2. Scope

### 2.1 In scope
- A single-page React app (`web/`) with: a Vietnamese query box, a lane selector
  (siglip2 / metaclip2, single or RRF), a virtualised ranked keyframe grid using
  thumbnails (ADR-0015), a frame-detail view (full image + `video_id`/`frame_id`,
  `youtube_url`, `description`, OCR/ASR snippet when available), and an in-UI
  "report issue" control.
- A typed API client and WebSocket client mirroring the SPEC-0026 schemas.
- Client-side screenshot capture for the issue report (browser canvas of the
  current UI), bundled with the query + lanes + returned frame ids + timestamp.
- Zustand store for query/result/selection state; TanStack Query for the
  REST fetches that are not on the hot path (frame detail).

### 2.2 Out of scope
- **TRAKE staging tray**, **QA answer panel**, **Ad-hoc** flows (proposal 06
  SS 3.1, SS 6).
- **Planner JSON panel** (proposal 06 SS 3.2): no planner LLM in the MVP.
- **Submission verification bar** (proposal 06 SS 3.7): no DRES submission in the
  MVP (SPEC-0018 deferred).
- **Keyframe scrubber / low-GOP video preview** (proposal 06 SS 3.5): the MVP
  shows prev/next neighbour thumbnails only.
- **Novice mode, hotkey overlay tutor, i18n toggle** (proposal 06 SS 7, SS 11):
  Vietnamese-default UI strings only.
- **Operator-trace logging** (SPEC-0019).
- All backend logic (SPEC-0026 owns query, image serving, and issue capture).

## 3. API contract / interface

```typescript
// web/src/api/types.ts  (mirror of SPEC-0026 Pydantic models)
export type Lane = "siglip2" | "metaclip2";
export type FusionMode = "single" | "rrf";

export interface QueryRequest {
  query_vi: string;
  lanes: Lane[];
  top_k: number;          // default 48
  fusion: FusionMode;     // default "single"
  rrf_k?: number;         // default 60, used only when fusion === "rrf"
}

export interface RankedFrame {
  pk: string;
  video_id: string;
  frame_id: string;
  rank: number;
  score: number;
  thumb_url: string;
  full_url: string;
  per_lane: Partial<Record<Lane, number>>;
}

export interface QueryResponse {
  query_vi: string;
  lanes: Lane[];
  fusion: FusionMode;
  results: RankedFrame[];
  took_ms: number;
}

export interface FrameDetail {
  pk: string;
  video_id: string;
  frame_id: string;
  frame_idx: number;
  youtube_url: string | null;
  description: string | null;
  od_tags: string[];
  ocr_text: string | null;
  asr_text: string | null;
  full_url: string;
  neighbours: string[];
}

export interface IssueReport {
  query_vi: string;
  lanes: Lane[];
  fusion: FusionMode;
  returned_frame_ids: string[];   // pks shown when the report was filed
  screenshot_png_b64: string;
  client_timestamp: string;       // ISO-8601
  note?: string;
}
```

```typescript
// web/src/api/client.ts
export interface ApiClient {
  // REST fallback + non-hot-path fetches
  query(req: QueryRequest): Promise<QueryResponse>;
  frameDetail(pk: string): Promise<FrameDetail>;
  reportIssue(report: IssueReport): Promise<{ issue_url: string | null }>;
}

// web/src/api/ws.ts
export interface QueryChannel {
  // hot path: send a QueryRequest, resolve on the QueryResponse frame (ADR-0004)
  send(req: QueryRequest): Promise<QueryResponse>;
  onError(handler: (message: string) => void): void;
}

// web/src/store.ts  (Zustand)
export interface UiState {
  query: string;
  lanes: Lane[];
  fusion: FusionMode;
  results: RankedFrame[];
  selectedPk: string | null;
  detail: FrameDetail | null;
  status: "idle" | "loading" | "error";
  tookMs: number | null;
}
```

## 4. Behaviour

- **Run a query**: the operator types Vietnamese text and submits (Enter or the
  search button); the app sends a `QueryRequest` over the WebSocket channel and
  renders the returned `results` in the grid, with `took_ms` shown. An empty
  query is not sent (the button is disabled / Enter is a no-op).
- **Lane selection**: the operator picks one lane (single) or two lanes
  (switching `fusion` to `rrf`); the choice is sent in the next `QueryRequest`.
- **Grid render**: results render as a virtualised thumbnail grid - only visible
  thumbnails mount (react-window or react-virtuoso, ADR-0004) - using each
  `RankedFrame.thumb_url`. Empty results show an explicit "no results" state, not
  a blank grid.
- **Frame detail**: clicking a thumbnail selects it and loads `FrameDetail` via
  REST; the detail view shows the full image (`full_url`), `video_id`/`frame_id`,
  `youtube_url` link, `description`, OCR/ASR snippet when present (hidden when
  `null`), and prev/next neighbour thumbnails.
- **Report an issue**: the "report issue" control captures a PNG screenshot of
  the current UI (browser canvas), bundles it with the current query, lanes,
  fusion, the `pk`s currently shown, and an ISO-8601 client timestamp into an
  `IssueReport`, optionally with a free-text note, and POSTs it to
  `/api/issues`. On success it shows the returned issue URL (or a "saved locally"
  message on fallback). The control is reachable without leaving the grid.
- **WebSocket loss**: if the channel drops, the client shows a reconnecting state
  and falls back to the REST `query` endpoint so a query still completes.
- **Error**: a backend error (422 / 5xx / WS error frame) surfaces a visible
  message; the app never silently shows stale results as if fresh.

## 5. Acceptance criteria

- **AC1**: submitting a non-empty Vietnamese query sends one `QueryRequest` with
  the selected lanes/fusion and renders the returned `results` in rank order.
  Verified in `web/tests/query_flow.test.ts` (mocked channel).
- **AC2**: an empty/whitespace query does not issue a request and keeps the prior
  grid. Verified in `web/tests/empty_query.test.ts`.
- **AC3**: the grid is virtualised - with 200 results, the number of mounted
  thumbnail nodes is bounded (does not equal 200). Verified in
  `web/tests/grid_virtualisation.test.ts`.
- **AC4**: clicking a thumbnail loads and shows `FrameDetail` with
  `video_id`/`frame_id` and the full image; OCR/ASR rows are hidden when the
  fields are `null`. Verified in `web/tests/frame_detail.test.ts`.
- **AC5**: the lane selector changes the `lanes`/`fusion` sent on the next query
  (single -> rrf when a second lane is chosen). Verified in
  `web/tests/lane_select.test.ts`.
- **AC6**: the "report issue" control produces an `IssueReport` carrying the
  current query, lanes, fusion, the shown `pk`s, a non-empty
  `screenshot_png_b64`, and an ISO-8601 `client_timestamp`, and POSTs it to
  `/api/issues`. Verified in `web/tests/issue_capture.test.ts` (mocked POST).
- **AC7**: on a WebSocket error the app shows a reconnecting state and completes
  the next query via the REST fallback. Verified in `web/tests/ws_fallback.test.ts`.

## 6. Non-functional requirements

- **Initial load**: time-to-interactive < 500 ms on the shared-server LAN
  (proposal 06 SS 9).
- **Grid render**: time to first thumbnail painted < 300 ms after a
  `QueryResponse`; per-thumbnail render p50 < 16 ms; scroll at 60 FPS with the
  virtualised grid (proposal 06 SS 9).
- **Thumbnail count**: render up to 48 thumbnails per page (8x6, proposal 06
  SS 2) and virtualise up to the `top_k` ceiling of 500 without mounting all
  nodes.
- **Hotkey/UI response**: query submit and selection respond in < 50 ms
  (excluding the network round-trip) (proposal 06 SS 9).
- **Compatibility**: modern evergreen browsers (Chromium-based primary, matching
  the finals laptop); no IE/legacy support.

## 7. Dependencies

- **Internal**: SPEC-0026 (the API: query, frame detail, image URLs, issue
  capture). Proposal 06 is the design reference; ADR-0004 the stack mandate;
  ADR-0015 the thumbnail URL scheme.
- **External (net-new frontend toolchain, added in the implementing PR)**:
  `react@18`, `vite`, `typescript`, `zustand`, `@tanstack/react-query`,
  `tailwindcss`, `shadcn/ui`, a grid virtualiser (`react-window` or
  `react-virtuoso`), and a DOM-to-canvas screenshot library (e.g. `html2canvas`)
  for the issue capture. These are recorded here per AGENTS.md; the exact
  virtualiser and screenshot lib are confirmed at approval (SS 9).
- **Data**: none directly; all data comes from the SPEC-0026 API and the static
  image tiers.

## 8. Test plan

- **Component / unit** (`web/tests/`, Vitest + Testing Library, mocked API):
  `query_flow`, `empty_query`, `grid_virtualisation`, `frame_detail`,
  `lane_select`, `issue_capture`, `ws_fallback` (the AC1-AC7 tests above).
- **E2e** (Playwright, against a SPEC-0026 instance backed by Milvus Lite +
  DummyEmbedder fixtures): search -> grid -> open detail -> file an issue. Mirrors
  proposal 06 SS 12; the MVP flow omits submit/verify (no DRES).
- **Manual smoke**: load the served URL, run three Vietnamese queries, open a few
  frames, file one issue, confirm it appears in GitHub.

## 9. Open questions

- **Q1**: Grid virtualiser choice (`react-window` vs `react-virtuoso`). Both meet
  the budget; recommend `react-virtuoso` for simpler variable-size handling.
  Confirm at approval.
- **Q2**: Screenshot library. `html2canvas` is the common choice but can be heavy
  and imperfect on some CSS; an alternative is the browser
  `getDisplayMedia`/`captureStream` path (requires a user gesture). Recommend
  `html2canvas` for zero extra prompts; confirm.
- **Q3**: Whether the MVP keeps a minimal query-history list (proposal 06 SS 3.6)
  or defers it. Recommend a lightweight last-10 in-memory list; confirm scope.
- **Q4**: Whether OCR/ASR snippets should be shown at all in the MVP given
  SPEC-0005 has not landed (the fields will be `null` on the proxy). Recommend
  rendering the rows conditionally now so no UI change is needed when SPEC-0005
  lands.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-06-05 | spec author (AI, user-directed) | Created (Draft). MVP React + WebSocket KIS console (query box, lane selector, virtualised thumbnail grid, frame detail, in-UI issue capture) as the proposal-06 subset, served by the SPEC-0026 API (ADR-0004 stack, ADR-0013 shared server, ADR-0015 thumbnails). TRAKE/QA/planner/verification-bar/novice-mode out of scope. Awaiting human approval before code. |
| 2026-06-07 | team lead (approval) | Status Draft -> Approved (human approval gate per AGENTS.md, PR #24). Q1 resolved: grid virtualiser = `react-virtuoso`. Q2 resolved: screenshot lib = `html2canvas`. Q3 resolved: keep a lightweight in-memory last-10 query history. Q4 resolved: render OCR/ASR rows conditionally now (null on proxy). Implementation begins on `spec/0026-mvp-serving-api`. |
