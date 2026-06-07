# Implements SPEC-0026 SS 5 AC3 (empty / blank query is 422, no search).
"""AC3: `query_vi=""` (or whitespace) returns 422 and runs no search."""

from __future__ import annotations


def test_empty_query_is_422_AC3(serving_client) -> None:
    # min_length=1 -> pydantic rejects an empty string before the service.
    resp = serving_client.post("/api/query", json={"query_vi": "", "lanes": ["siglip2"]})
    assert resp.status_code == 422, resp.text


def test_whitespace_query_is_422_AC3(serving_env) -> None:
    # A whitespace-only query passes pydantic min_length but must be rejected by
    # the service *before* any store search runs.
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from aic2026.serving.app import create_app

    app = create_app(serving_env.config, service=serving_env.service)
    with (
        TestClient(app) as client,
        patch.object(serving_env.store, "search", wraps=serving_env.store.search) as spy,
    ):
        resp = client.post("/api/query", json={"query_vi": "   ", "lanes": ["siglip2"]})
        assert resp.status_code == 422, resp.text
        spy.assert_not_called()
