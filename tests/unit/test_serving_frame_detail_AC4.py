# Implements SPEC-0026 SS 5 AC4 (frame detail point lookup + 404).
"""AC4: `GET /api/frame/{pk}` returns a `FrameDetail` for an ingested pk with
video_id / frame_id / frame_idx matching the store; an unknown pk returns 404.
"""

from __future__ import annotations


def test_frame_detail_for_known_pk_AC4(serving_client) -> None:
    pk = "L25_V001_0005"
    resp = serving_client.get(f"/api/frame/{pk}")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["pk"] == pk
    assert body["video_id"] == "L25_V001"
    assert body["frame_id"] == "0005"
    assert body["frame_idx"] == 5
    assert body["full_url"] == "/frames/L25_V001/0005.jpg"
    # SPEC-0005 has not landed: OCR/ASR are null on the proxy.
    assert body["ocr_text"] is None
    assert body["asr_text"] is None
    # Neighbours are the adjacent frame_idx pks within the same video.
    assert set(body["neighbours"]) == {"L25_V001_0004", "L25_V001_0006"}


def test_frame_detail_unknown_pk_is_404_AC4(serving_client) -> None:
    resp = serving_client.get("/api/frame/L25_V999_9999")
    assert resp.status_code == 404, resp.text
