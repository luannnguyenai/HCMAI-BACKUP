# Implements SPEC-0026 SS 5 AC7 (readiness gate).
"""AC7: `/readyz` returns 503 when the thumbnail tier is empty (or the
collection is unloaded / row_count == 0) and 200 with a populated `ReadyStatus`
once the collection is loaded, row_count > 0, and the thumb tier is non-empty.
"""

from __future__ import annotations

_FAKE_JPEG = b"\xff\xd8\xff\xe0fake\xff\xd9"


def test_not_ready_when_thumbs_empty_AC7(serving_env, serving_client) -> None:
    # serving_env starts with an empty thumb_root -> not ready even though the
    # collection is loaded with rows.
    resp = serving_client.get("/readyz")
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["ready"] is False
    assert body["collection_loaded"] is True
    assert body["row_count"] == 20  # two videos x ten frames
    assert body["thumbnails_present"] is False
    assert sorted(body["lanes_available"]) == ["metaclip2", "siglip2"]


def test_ready_when_all_three_hold_AC7(serving_env, serving_client) -> None:
    # Hydrate the thumbnail tier; now all three readiness conditions hold.
    (serving_env.thumb_root / "L25_V001").mkdir(parents=True, exist_ok=True)
    (serving_env.thumb_root / "L25_V001" / "0000.jpg").write_bytes(_FAKE_JPEG)

    resp = serving_client.get("/readyz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ready"] is True
    assert body["collection_loaded"] is True
    assert body["row_count"] == 20
    assert body["thumbnails_present"] is True
