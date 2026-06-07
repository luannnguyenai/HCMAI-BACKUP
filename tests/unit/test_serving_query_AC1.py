# Implements SPEC-0026 SS 5 AC1 (single-lane KIS query response shape).
"""AC1: `POST /api/query` with a valid Vietnamese query and `lanes=[siglip2]`
returns a `QueryResponse` whose results are <= top_k, 1-based contiguous rank,
descending score, each with non-empty thumb_url / full_url.
"""

from __future__ import annotations


def test_single_lane_query_response_shape_AC1(serving_client) -> None:
    top_k = 5
    resp = serving_client.post(
        "/api/query",
        json={"query_vi": "mot nguoi dang chay tren duong", "lanes": ["siglip2"], "top_k": top_k},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["query_vi"] == "mot nguoi dang chay tren duong"
    assert body["lanes"] == ["siglip2"]
    assert body["fusion"] == "single"
    assert isinstance(body["took_ms"], (int, float))

    results = body["results"]
    assert 0 < len(results) <= top_k

    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(results) + 1)), "ranks must be 1-based and contiguous"

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"

    for r in results:
        assert r["thumb_url"], "thumb_url must be non-empty"
        assert r["full_url"], "full_url must be non-empty"
        assert r["pk"] == f"{r['video_id']}_{r['frame_id']}"
        assert r["thumb_url"] == f"/thumbs/{r['video_id']}/{r['frame_id']}.jpg"
        assert r["full_url"] == f"/frames/{r['video_id']}/{r['frame_id']}.jpg"
        assert r["per_lane"] == {"siglip2": r["score"]}


def test_query_caps_at_top_k_AC1(serving_client) -> None:
    # The store holds 20 rows; a top_k below that must truncate.
    resp = serving_client.post(
        "/api/query",
        json={"query_vi": "canh bien", "lanes": ["siglip2"], "top_k": 3},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["results"]) == 3
