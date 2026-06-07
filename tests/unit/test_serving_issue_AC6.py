# Implements SPEC-0026 SS 5 AC6 (issue capture: GitHub path + local fallback).
"""AC6: `POST /api/issues` returns an `IssueResponse`; with no GitHub
repo/token it writes the local fallback and returns `fallback_path` (never 5xx).
The GitHub path is exercised against a mocked HTTP transport.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from aic2026.serving.app import create_app
from aic2026.serving.config import ServingConfig
from aic2026.serving.issues import IssueSink
from aic2026.serving.models import FusionMode, IssueReport, Lane

_REPORT_JSON = {
    "query_vi": "nguoi dan ong mac ao do",
    "lanes": ["siglip2"],
    "fusion": "single",
    "returned_frame_ids": ["L25_V001_0000", "L25_V001_0001"],
    "screenshot_png_b64": base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode(),
    "client_timestamp": "2026-06-07T14:30:00Z",
    "note": "wrong frames returned",
}


def test_local_fallback_when_no_github_AC6(serving_env) -> None:
    # github_repo is None -> never the GitHub path, even if GITHUB_TOKEN is set
    # in the environment (as it is on GitHub Actions).
    sink = IssueSink(serving_env.config)  # github_repo None
    app = create_app(serving_env.config, service=serving_env.service, issue_sink=sink)
    with TestClient(app) as client:
        resp = client.post("/api/issues", json=_REPORT_JSON)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["issue_url"] is None
    assert body["fallback_path"], "a local fallback path must be returned"

    payload = json.loads(Path(body["fallback_path"]).read_text(encoding="utf-8"))
    assert payload["query_vi"] == _REPORT_JSON["query_vi"]
    assert "screenshot_png_b64" not in payload  # PNG is written as a sibling file


def test_github_path_with_mocked_transport_AC6(serving_env) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(201, json={"html_url": "https://github.com/o/r/issues/7"})

    uploaded: dict[str, object] = {}

    def fake_uploader(png: bytes, key: str) -> str:
        uploaded["key"] = key
        uploaded["len"] = len(png)
        return f"https://cdn.example/{key}"

    config = replace(serving_env.config, github_repo="o/r")
    sink = IssueSink(
        config,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        screenshot_uploader=fake_uploader,
        github_token="fake-token",
    )
    report = IssueReport(
        query_vi="test",
        lanes=[Lane.siglip2],
        fusion=FusionMode.single,
        returned_frame_ids=["L25_V001_0000"],
        screenshot_png_b64=base64.b64encode(b"\x89PNGfake").decode(),
        client_timestamp="2026-06-07T14:30:00Z",
        note="github path",
    )
    resp = sink.capture(report)

    assert resp.issue_url == "https://github.com/o/r/issues/7"
    assert resp.fallback_path is None
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.github.com/repos/o/r/issues"
    assert captured["auth"] == "Bearer fake-token"
    assert "test" in captured["body"]["title"]
    assert uploaded["key"].startswith("issues/")
    assert uploaded["key"].endswith("screenshot.png")
    # The uploaded screenshot link is embedded in the issue body.
    assert "https://cdn.example/" in captured["body"]["body"]


def test_github_failure_falls_back_locally_AC6(serving_env) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "server error"})

    config: ServingConfig = replace(serving_env.config, github_repo="o/r")
    sink = IssueSink(
        config,
        http_client=httpx.Client(transport=httpx.MockTransport(boom)),
        screenshot_uploader=lambda png, key: None,
        github_token="fake-token",
    )
    report = IssueReport(
        query_vi="test",
        lanes=[Lane.siglip2],
        fusion=FusionMode.single,
        returned_frame_ids=[],
        screenshot_png_b64=base64.b64encode(b"x").decode(),
        client_timestamp="2026-06-07T14:30:00Z",
    )
    resp = sink.capture(report)
    # A failed GitHub round-trip must not lose the report.
    assert resp.issue_url is None
    assert resp.fallback_path is not None
