---
id: SPEC-0026
title: MVP serving API (Vietnamese KIS query + keyframe image serving + issue capture)
status: Approved
owner: unassigned
created: 2026-06-05
updated: 2026-06-07
implements_proposal: docs/proposals/01-interactive-system-architecture.md SS 5.1
related_adrs:
  - ADR-0003
  - ADR-0004
  - ADR-0011
  - ADR-0013
  - ADR-0014
  - ADR-0015
depends_on:
  - SPEC-0001
  - SPEC-0004
  - SPEC-0006
---

# SPEC-0026 - MVP serving API

> The backend HTTP + WebSocket service that turns a Vietnamese KIS text query
> into a ranked list of keyframes by wrapping the merged SPEC-0006
> `MilvusBackend`, serves the keyframe thumbnail and full images as static
> files, captures in-UI issue reports to GitHub, and reports health/readiness.
> It is the single shared server's API surface (ADR-0013); the React UI in
> SPEC-0027 is its only client. KIS only; QA / TRAKE / Ad-hoc / the automatic
> agent track / C2 learned fusion are out of scope.

## 1. Context

The team needs to qualitatively try Vietnamese KIS retrieval (text query ->
ranked keyframes) on the AIC2025 proxy corpus (121,457 keyframes / 546 videos,
no ground-truth answer key) before the June 25 corpus lands. SPEC-0006 is merged
and validated on Milvus standalone; this spec puts a thin web service in front
of it so a browser can drive it. [ADR-0013](../adr/ADR-0013-mvp-single-shared-server-from-r2.md)
fixes the topology (one shared server fed from R2);
[ADR-0014](../adr/ADR-0014-mvp-reuse-milvus-backend-standalone.md) fixes the
retrieval path (reuse `MilvusBackend` on Milvus standalone, not a numpy path);
[ADR-0015](../adr/ADR-0015-keyframe-image-hosting-thumbnails.md) fixes image
serving (pre-generated thumbnail tier, static serving). The web framework is
FastAPI and the transport is WebSocket per [ADR-0004](../adr/ADR-0004-no-streamlit-react-websocket-ui.md);
this spec does not re-decide either. Online query encoding (text tower) runs per
[ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md); image towers never
run online.

## 2. Scope

### 2.1 In scope
- A FastAPI service exposing: a query endpoint (REST `POST /api/query` and a
  `WS /ws` channel), a frame-detail endpoint, static keyframe thumbnail + full
  image serving, an issue-capture endpoint, and health/readiness endpoints.
- A query path: Vietnamese text -> text-tower encode (siglip2 / metaclip2 via
  the SPEC-0004 `Embedder`) -> per-lane `MilvusKeyframeStore.search` -> ranked
  `frame_id` + `score` lists -> single-lane passthrough or RRF combine
  ([ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md)) -> `QueryResponse`.
- Lane selection (request chooses one or more of the online lanes; default
  defined in SS 4).
- A provision-time startup contract that ingests the SPEC-0006 indexes from R2
  into Milvus standalone and hydrates the image tiers (ADR-0013 / ADR-0015).
- Pydantic request/response models and endpoint contracts (this spec defines
  shapes and behaviour only).

### 2.2 Out of scope
- **QA, TRAKE, Ad-hoc** task types and the **automatic-track agent** (SPEC-0017).
  The query endpoint accepts only KIS free-text.
- **C2 learned fusion** (SPEC-0015): ground-truth-blocked; the MVP fuses by
  single-lane or RRF only.
- **DRES submission** (SPEC-0018): no competition submission in the MVP; testing
  is qualitative.
- **The query encoder internals** (SPEC-0004) and **the Milvus store internals**
  (SPEC-0006): this spec wraps them, it does not reimplement them.
- **Thumbnail generation** (the offline pass, ADR-0015): this spec consumes the
  banked thumbnails; it does not produce them.
- **The React UI** (SPEC-0027).
- **Authentication / multi-tenant access control** beyond a single shared-secret
  gate (SS 9 Q1).

## 3. API contract / interface

```python
# aic2026/serving/models.py
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field

class Lane(StrEnum):
    siglip2 = "siglip2"
    metaclip2 = "metaclip2"

class FusionMode(StrEnum):
    single = "single"   # one lane, scores passed through
    rrf = "rrf"         # reciprocal-rank fusion over >= 2 lanes (ADR-0008)

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_vi: str = Field(min_length=1, max_length=512)
    lanes: list[Lane] = Field(default_factory=lambda: [Lane.siglip2])
    top_k: int = Field(default=48, ge=1, le=500)
    fusion: FusionMode = FusionMode.single
    rrf_k: int = Field(default=60, ge=1)          # honoured only when fusion == rrf

class RankedFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pk: str                       # SPEC-0006 global pk "<video_id>_<frame_id>"
    video_id: str
    frame_id: str
    rank: int = Field(ge=1)       # 1-based, post-fusion
    score: float                  # fused score (IP for single, RRF score for rrf)
    thumb_url: str                # static thumbnail path
    full_url: str                 # static full-image path
    per_lane: dict[Lane, float] = Field(default_factory=dict)  # pre-fusion lane scores

class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_vi: str
    lanes: list[Lane]
    fusion: FusionMode
    results: list[RankedFrame]
    took_ms: float                # server-side wall-clock for encode + search + fuse

class FrameDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pk: str
    video_id: str
    frame_id: str
    frame_idx: int
    youtube_url: str | None = None
    description: str | None = None
    od_tags: list[str] = Field(default_factory=list)
    ocr_text: str | None = None   # present only when SPEC-0005 lands; else None
    asr_text: str | None = None   # present only when SPEC-0005 lands; else None
    full_url: str
    neighbours: list[str] = Field(default_factory=list)  # prev/next pks in same video

class IssueReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_vi: str
    lanes: list[Lane]
    fusion: FusionMode
    returned_frame_ids: list[str]            # pks shown when the report was filed
    screenshot_png_b64: str                  # base64 PNG of the UI at report time
    client_timestamp: str                    # ISO-8601 from the browser
    note: str | None = Field(default=None, max_length=2000)

class IssueResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issue_url: str | None        # GitHub issue URL, or None on local-fallback
    fallback_path: str | None    # local path when GitHub is unavailable

class ReadyStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ready: bool
    collection_loaded: bool
    row_count: int
    thumbnails_present: bool
    lanes_available: list[Lane]
```

```python
# aic2026/serving/app.py  (FastAPI route signatures; ADR-0004)
def create_app(config: "ServingConfig") -> "FastAPI": ...

# GET  /healthz                      -> 200 {"status": "ok"}  (liveness)
# GET  /readyz                       -> ReadyStatus (200 when ready, 503 otherwise)
# POST /api/query   (QueryRequest)   -> QueryResponse
# WS   /ws          (QueryRequest msg) -> streams QueryResponse msg
# GET  /api/frame/{pk}               -> FrameDetail  (404 if unknown)
# POST /api/issues  (IssueReport)    -> IssueResponse
# GET  /thumbs/{video_id}/{frame_id}.jpg   -> image/jpeg (static; nginx in prod)
# GET  /frames/{video_id}/{frame_id}.jpg   -> image/jpeg (static; nginx in prod)
```

```python
# aic2026/serving/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ServingConfig:
    milvus_uri: str                 # http://127.0.0.1:19530 (standalone, ADR-0014)
    collection: str = "keyframes"
    online_lanes: tuple[str, ...] = ("siglip2", "metaclip2")
    thumb_root: Path = Path("/data/thumbs")   # hydrated from R2 (ADR-0015)
    full_root: Path = Path("/data/frames")
    github_repo: str | None = None  # "owner/repo" for issue capture; None -> local fallback
    shared_secret: str | None = None  # shared-secret gate, required in prod (SS 9 Q1 RESOLVED)
    encode_device: str = "cpu"      # text-tower device; MVP runs CPU (SS 9 Q4 RESOLVED)
```

## 4. Behaviour

- **Normal query (single lane)**: validate `QueryRequest`; encode `query_vi`
  once with the lane's text tower (SPEC-0004 `Embedder.encode_text`); call
  `MilvusKeyframeStore.search(field, q, top_k)`; map each `Hit` to a
  `RankedFrame` with `thumb_url`/`full_url` derived from `video_id`/`frame_id`
  (ADR-0015 key scheme); return `QueryResponse` with `took_ms`.
- **Normal query (multi-lane, fusion=rrf)**: encode once per requested lane,
  search each lane to `top_k`, combine by RRF with `rrf_k`
  ([ADR-0008](../adr/ADR-0008-rrf-as-runtime-fallback.md)), emit fused 1-based
  ranks and keep `per_lane` raw scores. Requesting `fusion=rrf` with one lane is
  a 422.
- **Default lane**: when `lanes` is omitted, query `siglip2` single-lane. This
  is the locked MVP default (SS 9 Q3 RESOLVED): one ranked list per query so
  testers can attribute issues to a single lane; `metaclip2` and `fusion=rrf`
  remain opt-in via the request.
- **Empty / blank query**: `query_vi` shorter than 1 non-whitespace character is
  a 422; no search runs.
- **WebSocket**: a client sends a `QueryRequest` JSON message; the server runs
  the same query path and sends back one `QueryResponse` message. Malformed
  messages get an error frame, not a dropped connection.
- **Frame detail**: `GET /api/frame/{pk}` reads scalar fields from the store for
  `pk`; unknown `pk` -> 404. `ocr_text` / `asr_text` are `None` until SPEC-0005
  lands; `neighbours` are the adjacent `frame_idx` pks in the same `video_id`.
- **Image serving**: thumbnails and full images are static files under
  `thumb_root` / `full_root` (nginx in production, a dev static route otherwise,
  ADR-0004); a missing file -> 404. Image bytes never pass through the query
  path.
- **Issue capture**: `POST /api/issues` writes a GitHub issue to `github_repo`
  from a fixed template (query, lanes, fusion, returned pks, client timestamp,
  note); the screenshot is uploaded to R2 (`issues/<timestamp>/screenshot.png`,
  ADR-0011) and linked in the body (SS 9 Q2). When `github_repo` or the token is
  absent, write the same payload to a local fallback file and return its path in
  `IssueResponse.fallback_path` (never 500 - a report must never be lost).
- **Startup / readiness**: on boot the service connects to Milvus standalone and
  loads the collection; `/readyz` is 200 only when the collection is loaded,
  `row_count > 0`, and `thumb_root` is non-empty; otherwise 503. Provisioning
  (ingest from R2 + image hydration) happens before the service is marked ready
  (ADR-0013).
- **KIS-only guard**: there is no task-type field; the endpoint serves free-text
  KIS retrieval only. QA/TRAKE/Ad-hoc are not exposed.

## 5. Acceptance criteria

- **AC1**: `POST /api/query` with a valid Vietnamese query and `lanes=[siglip2]`
  returns a `QueryResponse` whose `results` are `<= top_k`, 1-based contiguous
  `rank`, descending `score`, each with non-empty `thumb_url`/`full_url`.
  Verified in `tests/unit/test_serving_query_AC1.py` (Milvus Lite + DummyEmbedder).
- **AC2**: a request with two lanes and `fusion=rrf` returns RRF-fused results
  whose order matches a reference RRF over the per-lane lists, with `per_lane`
  populated; `fusion=rrf` with one lane returns 422. Verified in
  `tests/unit/test_serving_fusion_AC2.py`.
- **AC3**: `query_vi=""` (or whitespace) returns 422 and runs no search.
  Verified in `tests/unit/test_serving_validation_AC3.py`.
- **AC4**: `GET /api/frame/{pk}` returns a `FrameDetail` for an ingested pk with
  `video_id`/`frame_id`/`frame_idx` matching the store; an unknown pk returns
  404. Verified in `tests/unit/test_serving_frame_detail_AC4.py`.
- **AC5**: the thumbnail and full-image routes return `image/jpeg` for an
  existing keyframe and 404 for a missing one; the URL key scheme round-trips a
  `RankedFrame.pk` to the served file (ADR-0015). Verified in
  `tests/unit/test_serving_images_AC5.py`.
- **AC6**: `POST /api/issues` with a complete `IssueReport` returns an
  `IssueResponse`; with no GitHub repo/token configured it writes the local
  fallback and returns `fallback_path` (never 5xx). The GitHub path is exercised
  against a mocked HTTP transport. Verified in `tests/unit/test_serving_issue_AC6.py`.
- **AC7**: `/readyz` returns 503 before the collection is loaded / when
  `row_count == 0` / when `thumb_root` is empty, and 200 with a populated
  `ReadyStatus` once all three hold. Verified in `tests/unit/test_serving_ready_AC7.py`.
- **AC8**: a `WS /ws` client that sends one `QueryRequest` receives exactly one
  `QueryResponse` message conforming to the schema; a malformed message yields an
  error frame and the socket stays open. Verified in
  `tests/integration/test_serving_ws_AC8.py`.
- **AC9**: the query path encodes each query once per lane (no duplicate encode)
  and never loads an image tower (ADR-0003). Verified in
  `tests/unit/test_serving_encode_once_AC9.py` via a counting fake `Embedder`.

## 6. Non-functional requirements

- **Query latency**: `POST /api/query` server-side p95 < 1500 ms for one lane and
  < 2500 ms for two-lane RRF at `top_k=48` on the shared server (text-tower
  encode + per-lane ANN + fuse), excluding browser network. The MVP runs the
  text tower on **CPU** (SS 9 Q4 RESOLVED), so a single short-query encode (a few
  hundred ms to ~1 s) dominates: Milvus ANN p95 is only 16-51 ms on the validated
  proxy (SPEC-0006 SS 11.8). This CPU budget is acceptable for qualitative MVP
  testing; moving the text tower to a GPU (ADR-0003 finals path) restores the
  proposal-01 sub-800 ms target without code change (`encode_device="cuda"`).
- **Image-serve latency**: thumbnail p95 < 50 ms, full image p95 < 200 ms over
  the static tier (nginx), measured on the shared server LAN.
- **Concurrency**: >= 10 concurrent testers and >= 5 queries/sec aggregate with
  no query exceeding the latency NFR (async FastAPI, ADR-0004).
- **Issue capture**: `POST /api/issues` returns within p95 < 3 s including the
  GitHub round-trip; the local fallback path returns within p95 < 200 ms.
- **Compatibility**: Python 3.11+; FastAPI + uvicorn (async); `pymilvus >= 2.5`
  (the `index` extra); `httpx` (already a core dep) for the GitHub call;
  `boto3` (already a core dep) for the screenshot upload.

## 7. Dependencies

- **Internal**: SPEC-0006 (`MilvusBackend`, `MilvusKeyframeStore`,
  `hits_to_submissions`, `Hit`); SPEC-0004 (`Embedder` text tower for online
  encoding); SPEC-0001 (`Submission` shape reused by the SPEC-0006 adapter).
- **External**: `fastapi`, `uvicorn[standard]`, `websockets` (a new `serving`
  optional-dependency extra, to be added in the implementing PR per AGENTS.md
  "add deps in the spec"); `pymilvus[milvus_lite] >= 2.5` (existing `index`
  extra); `httpx >= 0.27`, `boto3 >= 1.34` (existing core deps).
- **Data**: the R2-banked SPEC-0006 indexes
  (`index/aic2025-proxy-3enc-20260604/`, ADR-0011), the R2 thumbnail tier
  (`thumbs/aic2025-proxy-3enc-20260604/`, ADR-0015), and the full-resolution
  keyframe JPGs. Organiser `youtube_url` / `description` / OD tags are sparse on
  the proxy and arrive with the June 25 corpus.

## 8. Test plan

- **Unit tests** (`tests/unit/`, Milvus Lite + DummyEmbedder, CPU/offline):
  - `test_serving_query_AC1.py`, `test_serving_fusion_AC2.py`,
    `test_serving_validation_AC3.py`, `test_serving_frame_detail_AC4.py`,
    `test_serving_images_AC5.py`, `test_serving_issue_AC6.py` (mocked GitHub
    transport + `moto` for the R2 screenshot upload), `test_serving_ready_AC7.py`,
    `test_serving_encode_once_AC9.py`.
- **Integration tests** (`tests/integration/`):
  - `test_serving_ws_AC8.py` - a WebSocket round-trip via the FastAPI test client.
  - Lease-box smoke (not CI): point the service at the live Milvus standalone +
    the hydrated image tiers; record query p95 and image-serve p95 in SS 10.
- **Manual smoke**: open `/readyz`, run a query via `POST /api/query`, open a
  returned `thumb_url` and `full_url`, file one issue, confirm the GitHub issue.

## 9. Open questions

- **Q1 RESOLVED (2026-06-05, user-directed)**: Access control is a single
  shared-secret gate (`shared_secret` in `ServingConfig`, sent as a request
  header); required in production. SSH-tunnel-only and open-private-network were
  rejected as too much per-tester setup / too little control for a shared URL.
- **Q2 (open)**: Screenshot handling for the GitHub issue. Recommend upload to R2
  and link (GitHub's API has no clean base64-attachment path); confirm the R2
  prefix `issues/<timestamp>/` and whether the bucket is readable for issue
  viewers, or whether a collapsed base64 block in the issue body is preferred.
  Resolve at implementation; does not block approval.
- **Q3 RESOLVED (2026-06-05, user-directed)**: Default served lane is `siglip2`
  single-lane (one ranked list per query, clean issue attribution); `metaclip2`
  and `fusion=rrf` remain opt-in via the request. Two-lane-RRF-by-default was
  rejected for the first test window (mixes signals when triaging issues).
- **Q4 RESOLVED (2026-06-05, user-directed)**: The online text tower runs on
  **CPU** on the shared server (no dedicated serving GPU required). The latency
  NFR in SS 6 is set to the CPU-encode-dominated budget; `encode_device` is a
  config knob so a GPU box can tighten it later without code change.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-06-05 | spec author (AI, user-directed) | Created (Draft). FastAPI + WebSocket service wrapping the merged SPEC-0006 MilvusBackend: KIS query (text -> per-lane ANN -> single/RRF), frame detail, static thumbnail + full image serving (ADR-0015), issue capture to GitHub, health/readiness, R2 startup contract (ADR-0013). KIS-only; QA/TRAKE/agent/C2 deferred. Awaiting human approval before code. |
| 2026-06-05 | implementer (user-directed) | Resolved Q1 (access = shared-secret gate), Q3 (default lane = siglip2 single-lane), Q4 (text tower on CPU). Revised SS 6 query-latency NFR to the CPU-encode-dominated budget (p95 < 1500 ms single / < 2500 ms RRF); added `encode_device` to `ServingConfig`. Q2 (screenshot R2-link vs base64) left open for implementation (non-blocking). |
| 2026-06-07 | team lead (approval) | Status Draft -> Approved (human approval gate per AGENTS.md, PR #24). ADR-0013/0014/0015 accepted alongside. Implementation begins on `spec/0026-mvp-serving-api`. Q2 resolved at implementation: screenshot uploaded to R2 (`issues/<timestamp>/screenshot.png`) and linked in the GitHub issue body; local-fallback writes the PNG next to the JSON payload. |
