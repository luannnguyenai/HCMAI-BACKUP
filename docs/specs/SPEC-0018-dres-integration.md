---
id: SPEC-0018
title: DRES integration client (login, evaluation list, submit)
status: Draft
owner: unassigned
created: 2026-05-26
updated: 2026-05-26
implements_proposal: docs/proposals/05-evaluation-harness.md
related_adrs:
  - ADR-0009
  - ADR-0010
depends_on:
  - SPEC-0001
---

# SPEC-0018 — DRES integration client

> A minimal Python client for the DRES (Distributed Retrieval Evaluation Server) used by AIC HCMC. Provides login, evaluation listing, session management, and submit. Used by SPEC-0001 (eval harness) for local mock-DRES runs and by the live finals interactive backend for real submissions. The login + submit flow is borrowed from the 2025 baseline under [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md).

## 1. Context

The interactive track's submissions go to the **DRES** server (open source, <https://github.com/dres-dev/DRES>). The production HCMC AIC instance is at **<https://eventretrieval.oj.io.vn>** (URL discovered in the 2025 baseline; see [`docs/research-notes/05-baseline-2025-analysis.md`](../research-notes/05-baseline-2025-analysis.md) §3.1).

The 2025 baseline (`ThanhToan2111/AIC_2026` at commit `c3c3545`, `streamlit_api.py:122-200`) contains a working login + submit flow, but coupled to Streamlit's `session_state`. We extract it into a clean Python module so the same client can be driven by:
- The eval harness ([SPEC-0001](SPEC-0001-evaluation-harness.md)) against a *local* DRES Docker instance.
- The interactive backend at finals against the *production* DRES URL.
- The automatic-track agent ([proposal 02](../proposals/02-automatic-track-agent.md)) for unattended submission.

## 2. Scope

### 2.1 In scope
- HTTP client for the DRES v2 API surface used at AIC HCMC: `/api/v2/login`, `/api/v2/client/evaluation/list`, `/api/v2/submit`, `/api/v2/logout`.
- Session lifecycle: login ? session-id caching ? submit ? logout.
- Pydantic models for `DresSession`, `Evaluation`, `Submission`, `SubmissionResult`.
- Sync and async (httpx) variants of the client.
- Configurable base URL (production vs local mock-DRES).
- Submission for all four task types: KIS (image submission), QA (text answer), Ad-hoc (image submission), TRAKE (ordered list of frames).

### 2.2 Out of scope
- Hosting the DRES server itself (delegated to SPEC-0001's Docker Compose under proposal 05).
- Scoring logic on the client side (DRES owns scoring).
- Authentication beyond username/password (DRES v2 has no SSO).
- WebSocket subscription to the live leaderboard (Phase 4 deliverable; not gating this spec).

## 3. API contract

### 3.1 Models

```python
class DresConfig(BaseModel):
    base_url: str                                 # e.g. "https://eventretrieval.oj.io.vn"
    username: str
    password: SecretStr
    timeout_seconds: float = 10.0
    max_retries: int = 3

class DresSession(BaseModel):
    session_id: str                               # opaque cookie / header value
    username: str
    user_id: str
    role: Literal["PARTICIPANT", "ADMIN", "VIEWER"]
    expires_at: datetime | None                   # if DRES tells us; else None

class Evaluation(BaseModel):
    id: str
    name: str
    template_id: str | None = None
    type: Literal["SYNCHRONOUS", "ASYNCHRONOUS"]
    status: Literal["CREATED", "ACTIVE", "PAUSED", "ENDED"]

class SubmissionTarget(BaseModel):
    kind: Literal["KIS", "QA", "AD_HOC", "TRAKE"]
    # KIS / Ad-hoc: a single (video_id, frame_index_or_ms)
    # QA: a free-text answer string
    # TRAKE: ordered list of (video_id, frame_index_or_ms)
    media_item_id: str | None = None             # required for KIS/Ad-hoc
    start_ms: int | None = None
    end_ms: int | None = None
    answer_text: str | None = None               # required for QA
    trake_items: list[tuple[str, int]] | None = None   # required for TRAKE

class SubmissionResult(BaseModel):
    accepted: bool
    submission_id: str | None
    server_message: str | None
    server_status: Literal["CORRECT", "WRONG", "INDETERMINATE"] | None
    latency_ms: float
```

### 3.2 Client interface

```python
class DresClient:
    def __init__(self, config: DresConfig) -> None: ...

    # Lifecycle
    def login(self) -> DresSession: ...
    def logout(self) -> None: ...
    def is_session_valid(self) -> bool: ...

    # Discovery
    def list_evaluations(self) -> list[Evaluation]: ...
    def select_evaluation(self, evaluation_id: str) -> Evaluation: ...

    # Submission
    def submit(self, target: SubmissionTarget) -> SubmissionResult: ...

    # Convenience
    def submit_kis(self, video_id: str, frame_index: int) -> SubmissionResult: ...
    def submit_qa(self, answer: str) -> SubmissionResult: ...
    def submit_trake(self, frames: list[tuple[str, int]]) -> SubmissionResult: ...

class AsyncDresClient:
    """async/await variant; same surface, returns coroutines."""
    ...
```

### 3.3 Configuration

Loaded from environment + optional config file:
```
DRES_BASE_URL=https://eventretrieval.oj.io.vn
DRES_USERNAME=<team-username>
DRES_PASSWORD=<team-password>   # in .env, never committed
DRES_TIMEOUT=10
```

## 4. Behaviour

### 4.1 Login
1. POST `{base_url}/api/v2/login` with `{"username": ..., "password": ...}`.
2. On 200, parse `{sessionId, username, id, role}` and return a `DresSession`.
3. On 401/403, raise `DresAuthError` with the server message.
4. On network failure, retry with exponential backoff up to `max_retries`.

### 4.2 Session management
- The session-id is held in memory only. Never written to disk.
- `is_session_valid()` does a lightweight `GET /api/v2/client/evaluation/list` and treats a 200 as valid, 401 as invalid.
- The client auto-re-logs in once on a 401 from any submission call; if the re-login also fails, the error propagates.

### 4.3 Evaluation discovery
- `list_evaluations()` returns the full list. The harness or operator picks one and calls `select_evaluation`.
- `select_evaluation` stores the id locally on the client; subsequent `submit()` calls scope to it.

### 4.4 Submit
- Build the DRES v2 submission body per task type. KIS/Ad-hoc use `mediaItemName` + `start`/`end`; QA uses `text`; TRAKE submits an ordered array.
- POST `{base_url}/api/v2/submit?session={session_id}` with the body.
- Parse the response; map to `SubmissionResult`. `server_status` may be `"CORRECT"`, `"WRONG"`, or `"INDETERMINATE"` (the last for tasks under human-judged review).
- Always populate `latency_ms` measured locally from the moment `submit()` was called.

### 4.5 Failure modes
- **Network timeout**: retry per `max_retries`; if all fail, return `SubmissionResult(accepted=False, server_message="network_timeout", ...)`.
- **5xx server error**: retry once after 1 s backoff; if still failing, return `accepted=False`.
- **400 malformed**: do not retry; raise `DresBadRequest`.
- **401 expired session**: silently re-login once; retry the original submit; if the retry 401s, raise `DresAuthError`.
- **Idempotency**: the client does *not* dedupe submissions. The caller is responsible (typically: the submission-verification UI bar from [SPEC-0013](SPEC-0013-submission-verification-panel.md)).

### 4.6 Logging
- One INFO log per `submit()`: `{task_id?, kind, latency_ms, accepted, server_status}`.
- DEBUG log of the full request body for failed submissions, with password redacted.

## 5. Acceptance criteria

- **AC1**: `DresClient(DresConfig(base_url=..., username=..., password=...))` instantiation succeeds without network calls.
- **AC2**: `client.login()` against a local DRES Docker instance returns a `DresSession` with non-empty `session_id`.
- **AC3**: `client.list_evaluations()` returns a `list[Evaluation]` of at least 1 item after login on a local DRES with a seeded task set.
- **AC4**: `client.submit_kis("video_001", 1234)` against an active evaluation produces a `SubmissionResult` with `accepted=True` and a finite `latency_ms`.
- **AC5**: `client.submit_qa("answer text")` produces a valid `SubmissionResult` with the QA body shape DRES expects.
- **AC6**: `client.submit_trake([("v1", 100), ("v1", 200), ("v1", 300), ("v1", 400)])` submits an ordered 4-frame TRAKE response.
- **AC7**: When DRES returns 401 mid-session, the client re-logs in once and retries; if the retry also 401s, the call raises `DresAuthError`.
- **AC8**: When DRES times out (configurable), the client retries per `max_retries` and on terminal failure returns `accepted=False` with `server_message="network_timeout"`.
- **AC9**: Passwords are never logged. A DEBUG log of a failed-submission request body redacts the `password` field if present.
- **AC10**: `AsyncDresClient` exposes the same surface and returns coroutines; an equivalent test runs against a mocked async server.

## 6. Non-functional requirements

- **Latency**: per-submission round-trip p95 < 500 ms on Vietnamese 5G to the production DRES URL. p50 < 200 ms on local DRES.
- **Reliability**: 99.0 % success rate on a 200-call smoke test against local DRES.
- **Memory**: client object < 10 MB resident; no per-call growth.
- **Compatibility**: Python 3.11+. `httpx >= 0.27`, `pydantic >= 2`.
- **Security**: passwords loaded from environment via `pydantic.SecretStr`; never written to disk; redacted in all logs.
- **Cost**: zero (DRES is self-hosted; no third-party API charges).

## 7. Dependencies

- **Internal**:
  - [SPEC-0001](SPEC-0001-evaluation-harness.md) — the eval harness imports `DresClient` to drive `bin/eval` submissions.
- **External**:
  - `httpx >= 0.27` (sync + async HTTP client)
  - `pydantic >= 2`
  - `tenacity >= 8` (retry policy)
  - `respx >= 0.20` (test-time HTTP mocking; dev dependency only)
- **Borrowed prior art** (per [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md)):
  - Login flow logic adapted from `ThanhToan2111/AIC_2026:streamlit_api.py:122-200` at commit `c3c3545`.
  - Submission body shape adapted from the same source, lines 200-380.
  - Once the borrow lands, it appears in the top-level `THIRD_PARTY.md`.

## 8. Test plan

### 8.1 Unit tests (`tests/unit/test_dres_client.py`)
- `test_instantiation_no_network_AC1`
- `test_login_success_against_local_dres_AC2` (uses `respx` mock)
- `test_list_evaluations_returns_at_least_one_AC3`
- `test_submit_kis_accepted_AC4`
- `test_submit_qa_body_shape_AC5`
- `test_submit_trake_ordered_AC6`
- `test_401_triggers_relogin_then_retry_AC7`
- `test_timeout_retries_then_fails_with_network_timeout_AC8`
- `test_password_never_logged_AC9`
- `test_async_client_surface_parity_AC10`

### 8.2 Integration tests (`tests/integration/test_dres_local.py`)
- Boot DRES via `testcontainers` (or attach to a running Docker-Compose instance). Run the full lifecycle: login ? list ? select ? submit KIS / QA / TRAKE ? logout. Assert each submission is reflected in DRES's `/api/v2/evaluation/{id}/state`.

### 8.3 Live smoke (optional, gated by env)
- `pytest --live-dres tests/live/test_dres_smoke.py` against the production URL with valid credentials. Default skip in CI; ran by hand by the operator during finals prep (Phase 4).

## 9. Open questions

- **Q1**: Does the production DRES at `eventretrieval.oj.io.vn` use the DRES v2 endpoints exactly as the open-source project documents, or does AIC HCMC run a fork? Need to confirm — possibly by asking the original baseline author (`ThanhToan2111`).
- **Q2**: TRAKE submission body — is the ordered list a JSON array of `{mediaItemName, start, end}` objects, or a flat list of media-item IDs? The 2025 baseline did not implement TRAKE submission cleanly (`streamlit_api.py` showed only KIS/QA). Need to verify against the DRES v2 spec.
- **Q3**: Session-id transport — header (`X-Session`), cookie (`SESSIONID=`), or query param (`?session=`)? The 2025 baseline used the query param (`?session={id}`); we should confirm this is preferred and not just one of several supported forms.
- **Q4**: Rate limiting — does DRES rate-limit submissions per session, and if so, at what rate? Affects our automatic-track agent's retry policy.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-26 | team lead | Created (Draft); seeded from 2025 baseline analysis |
