# Implements SPEC-0026 SS 3-4 (FastAPI app: routes, WebSocket, static images).
"""The FastAPI application factory for the MVP serving API (ADR-0004).

`create_app(config)` builds the HTTP + WebSocket surface over a `QueryService`
(the SPEC-0006 store + SPEC-0004 text towers) and an `IssueSink`. The service
and sink are injectable so the AC tests drive the app in-process against Milvus
Lite + `DummyEmbedder` with no GPU, no Docker, and no network.

Endpoints (SPEC-0026 SS 3):
  GET  /healthz                          liveness
  GET  /readyz                           ReadyStatus (200 ready / 503 not)
  POST /api/query        (QueryRequest)  QueryResponse
  WS   /ws               (QueryRequest)  streams QueryResponse
  GET  /api/frame/{pk}                   FrameDetail (404 unknown)
  POST /api/issues       (IssueReport)   IssueResponse
  GET  /thumbs/{video_id}/{frame_id}.jpg image/jpeg (static; nginx in prod)
  GET  /frames/{video_id}/{frame_id}.jpg image/jpeg (static; nginx in prod)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from aic2026.serving.config import ServingConfig
from aic2026.serving.issues import IssueSink
from aic2026.serving.models import (
    FrameDetail,
    IssueReport,
    IssueResponse,
    QueryRequest,
    QueryResponse,
)
from aic2026.serving.service import QueryService, QueryValidationError, build_default_service

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

logger = logging.getLogger(__name__)

SECRET_HEADER: str = "X-AIC-Secret"
_WS_POLICY_VIOLATION: int = 1008  # RFC 6455 close code for an auth failure.


def _make_secret_gate(config: ServingConfig) -> Callable[[str | None], None]:
    """A FastAPI dependency enforcing the shared-secret header when configured."""

    async def gate(
        x_aic_secret: Annotated[str | None, Header(alias=SECRET_HEADER)] = None,
    ) -> None:
        if config.shared_secret is not None and x_aic_secret != config.shared_secret:
            raise HTTPException(status_code=401, detail="invalid or missing shared secret")

    return gate


def _serve_image(root: Path, video_id: str, frame_id: str) -> FileResponse:
    """Static image serving with path-traversal containment (404 on miss)."""
    root_resolved = root.resolve()
    candidate = (root_resolved / video_id / f"{frame_id}.jpg").resolve()
    if root_resolved not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(candidate, media_type="image/jpeg")


def create_app(
    config: ServingConfig,
    *,
    service: QueryService | None = None,
    issue_sink: IssueSink | None = None,
) -> FastAPI:
    """Build the FastAPI app. Injects `service`/`issue_sink` for tests."""
    svc = service if service is not None else build_default_service(config)
    sink = issue_sink if issue_sink is not None else IssueSink(config)
    gate = _make_secret_gate(config)
    guarded = [Depends(gate)]

    app = FastAPI(title="AIC2026 MVP serving API (SPEC-0026)", version="0.1.0")
    app.state.config = config
    app.state.service = svc
    app.state.issue_sink = sink

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        status = await run_in_threadpool(svc.readiness)
        code = 200 if status.ready else 503
        return JSONResponse(status_code=code, content=status.model_dump(mode="json"))

    @app.post("/api/query", response_model=QueryResponse, dependencies=guarded)
    async def query(req: QueryRequest) -> QueryResponse:
        try:
            return await run_in_threadpool(svc.query, req)
        except QueryValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/frame/{pk}", response_model=FrameDetail, dependencies=guarded)
    async def frame(pk: str) -> FrameDetail:
        detail = await run_in_threadpool(svc.frame_detail, pk)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown pk {pk!r}")
        return detail

    @app.post("/api/issues", response_model=IssueResponse, dependencies=guarded)
    async def issues(report: IssueReport) -> IssueResponse:
        return await run_in_threadpool(sink.capture, report)

    @app.get("/thumbs/{video_id}/{frame_id}.jpg", dependencies=guarded)
    async def thumb(video_id: str, frame_id: str) -> FileResponse:
        return _serve_image(config.thumb_root, video_id, frame_id)

    @app.get("/frames/{video_id}/{frame_id}.jpg", dependencies=guarded)
    async def full(video_id: str, frame_id: str) -> FileResponse:
        return _serve_image(config.full_root, video_id, frame_id)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        if config.shared_secret is not None:
            provided = websocket.headers.get(SECRET_HEADER.lower()) or websocket.query_params.get(
                "secret"
            )
            if provided != config.shared_secret:
                await websocket.close(code=_WS_POLICY_VIOLATION)
                return
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    req = QueryRequest.model_validate_json(raw)
                    resp = await run_in_threadpool(svc.query, req)
                except (ValidationError, QueryValidationError) as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue
                await websocket.send_text(resp.model_dump_json())
        except WebSocketDisconnect:
            return

    return app
