# Implements SPEC-0026 SS 5 AC2 (two-lane RRF fusion + single-lane-rrf 422).
"""AC2: two lanes + fusion=rrf returns RRF-fused results whose order matches a
reference RRF over the per-lane lists, with `per_lane` populated; `fusion=rrf`
with one lane returns 422.
"""

from __future__ import annotations

import numpy as np

from aic2026.serving.models import Lane


def _reference_rrf(env, query_vi: str, top_k: int, rrf_k: int) -> list[str]:
    """Independent RRF over the two lanes' raw store hits -> ordered pks."""
    fused: dict[str, float] = {}
    for lane in (Lane.siglip2, Lane.metaclip2):
        vec = np.asarray(env.encoders[lane].encode_text([query_vi]), dtype=np.float32)
        hits = env.store.search(lane.value, vec, top_k=top_k)[0]
        for rank, h in enumerate(hits, start=1):
            fused[h.pk] = fused.get(h.pk, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [pk for pk, _ in ordered[:top_k]]


def test_two_lane_rrf_matches_reference_AC2(serving_env, serving_client) -> None:
    query_vi = "hai nguoi dang noi chuyen"
    top_k = 8
    rrf_k = 60
    resp = serving_client.post(
        "/api/query",
        json={
            "query_vi": query_vi,
            "lanes": ["siglip2", "metaclip2"],
            "fusion": "rrf",
            "top_k": top_k,
            "rrf_k": rrf_k,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["fusion"] == "rrf"
    assert body["lanes"] == ["siglip2", "metaclip2"]

    got_pks = [r["pk"] for r in body["results"]]
    expected_pks = _reference_rrf(serving_env, query_vi, top_k, rrf_k)
    assert got_pks == expected_pks

    # per_lane carries the raw pre-fusion scores for each lane the doc appeared in.
    for r in body["results"]:
        assert r["per_lane"], "per_lane must be populated under RRF"
        assert set(r["per_lane"]).issubset({"siglip2", "metaclip2"})


def test_rrf_with_one_lane_is_422_AC2(serving_client) -> None:
    resp = serving_client.post(
        "/api/query",
        json={"query_vi": "mot chiec xe", "lanes": ["siglip2"], "fusion": "rrf"},
    )
    assert resp.status_code == 422, resp.text
