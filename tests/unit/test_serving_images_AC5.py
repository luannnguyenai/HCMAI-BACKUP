# Implements SPEC-0026 SS 5 AC5 (static thumbnail + full image serving).
"""AC5: the thumbnail and full-image routes return image/jpeg for an existing
keyframe and 404 for a missing one; the URL key scheme round-trips a
RankedFrame.pk to the served file (ADR-0015).
"""

from __future__ import annotations

# A 1x1 JPEG is unnecessary; the route serves bytes verbatim with image/jpeg.
_FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"


def test_thumb_and_full_round_trip_from_pk_AC5(serving_env, serving_client) -> None:
    video_id, frame_id = "L25_V001", "0003"
    (serving_env.thumb_root / video_id).mkdir(parents=True, exist_ok=True)
    (serving_env.full_root / video_id).mkdir(parents=True, exist_ok=True)
    (serving_env.thumb_root / video_id / f"{frame_id}.jpg").write_bytes(_FAKE_JPEG)
    (serving_env.full_root / video_id / f"{frame_id}.jpg").write_bytes(_FAKE_JPEG)

    # The URL scheme is derivable from a RankedFrame's video_id/frame_id (== pk).
    thumb = serving_client.get(f"/thumbs/{video_id}/{frame_id}.jpg")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert thumb.content == _FAKE_JPEG

    full = serving_client.get(f"/frames/{video_id}/{frame_id}.jpg")
    assert full.status_code == 200
    assert full.headers["content-type"] == "image/jpeg"
    assert full.content == _FAKE_JPEG


def test_missing_image_is_404_AC5(serving_client) -> None:
    assert serving_client.get("/thumbs/L25_V001/9999.jpg").status_code == 404
    assert serving_client.get("/frames/L25_V001/9999.jpg").status_code == 404
