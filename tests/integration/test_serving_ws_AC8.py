# Implements SPEC-0026 SS 5 AC8 (WebSocket query round-trip + error frame).
"""AC8: a `WS /ws` client that sends one `QueryRequest` receives exactly one
`QueryResponse`; a malformed message yields an error frame and the socket stays
open.

Self-contained (tests/integration does not see tests/unit/conftest.py): builds a
Milvus-Lite-backed `QueryService` inline, then drives it through fastapi's
in-process WebSocket test client. Skips when Milvus Lite cannot start.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _build_service(tmp_path: Path):
    pytest.importorskip("pymilvus", reason="pymilvus not installed (index extra)")
    pytest.importorskip("milvus_lite", reason="milvus-lite backend not installed")
    from pymilvus.exceptions import MilvusException

    from aic2026.embedding.dummy import DummyEmbedder
    from aic2026.embedding.extract import extract_image_embeddings
    from aic2026.index.milvus_schema import DenseField
    from aic2026.index.milvus_store import EncoderSource, MilvusKeyframeStore
    from aic2026.serving.config import ServingConfig
    from aic2026.serving.models import Lane
    from aic2026.serving.service import QueryService

    dim = 8
    fields = (DenseField("siglip2", dim), DenseField("metaclip2", dim))
    store = MilvusKeyframeStore(uri=str(tmp_path / "ws.db"), collection="keyframes", fields=fields)
    try:
        store.ensure_collection()
    except (RuntimeError, OSError, MilvusException) as exc:  # pragma: no cover - env skip
        pytest.skip(f"Milvus Lite could not initialise: {exc!r}")

    frame_ids = [f"{i:04d}" for i in range(10)]

    def src(field: str, video: str) -> EncoderSource:
        img_dir = tmp_path / field / video / "imgs"
        img_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for fid in frame_ids:
            p = img_dir / f"{fid}.jpg"
            p.write_bytes(f"{field}:{video}:{fid}".encode())
            paths.append(p)
        res = extract_image_embeddings(
            paths, DummyEmbedder(dim=dim, model_id=f"dummy-{field}"), out=tmp_path / field / video
        )
        return EncoderSource(vectors=res.vectors_path, manifest=res.manifest_path)

    for video in ("L25_V001", "L25_V002"):
        store.ingest({f.name: src(f.name, video) for f in fields}, video_id=video)

    encoders = {
        Lane.siglip2: DummyEmbedder(dim=dim, model_id="q-siglip2"),
        Lane.metaclip2: DummyEmbedder(dim=dim, model_id="q-metaclip2"),
    }
    thumb_root = tmp_path / "thumbs"
    thumb_root.mkdir()
    config = ServingConfig(
        milvus_uri=store.uri,
        collection=store.collection,
        thumb_root=thumb_root,
        full_root=tmp_path / "frames",
        issue_fallback_dir=tmp_path / "issues",
    )
    return QueryService(store, encoders, config), config


def _assert_query_response(payload: dict, lanes: Sequence[str]) -> None:
    assert payload["fusion"] in {"single", "rrf"}
    assert payload["lanes"] == list(lanes)
    assert isinstance(payload["results"], list)
    assert "took_ms" in payload


def test_ws_query_round_trip_and_error_frame_AC8(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aic2026.serving.app import create_app

    service, config = _build_service(tmp_path)
    app = create_app(config, service=service)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        # One valid QueryRequest -> exactly one QueryResponse.
        ws.send_text(json.dumps({"query_vi": "mot con cho", "lanes": ["siglip2"], "top_k": 4}))
        resp = json.loads(ws.receive_text())
        _assert_query_response(resp, ["siglip2"])
        assert len(resp["results"]) <= 4

        # A malformed message yields an error frame, not a dropped connection.
        ws.send_text("{ this is not valid json")
        err = ws.receive_json()
        assert "error" in err

        # The socket is still open: a subsequent valid query still works.
        ws.send_text(json.dumps({"query_vi": "bien xanh", "lanes": ["siglip2"], "top_k": 2}))
        resp2 = json.loads(ws.receive_text())
        _assert_query_response(resp2, ["siglip2"])
