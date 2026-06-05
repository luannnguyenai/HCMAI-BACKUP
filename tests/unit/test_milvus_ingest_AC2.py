# Proves SPEC-0006 AC2: ingest aligns .npy rows to manifest rows by the
# per-video frame_id, derives video_id from the SOURCE (the npy filename, not
# parsed from frame_id), composes the global primary key `<video_id>_<frame_id>`,
# sets frame_idx 0-based within the video, and IngestResult.n_rows equals the
# row count.

from __future__ import annotations

from collections.abc import Callable

from aic2026.index.milvus_schema import DenseField
from aic2026.index.milvus_store import MilvusKeyframeStore

from .conftest import make_encoder_source

# Small dims keep the Lite fixtures fast; the schema is dim-generic.
_FIELDS = (
    DenseField("siglip2", 8),
    DenseField("metaclip2", 4),
    DenseField("qwen3vl", 6, online_query=False),
)


def _ingest_video(store, tmp_path, video, frame_ids):
    """Ingest one video the way bin/index ingest-all does: per-video sources,
    video_id derived from the `<video>.npy` filename (not passed explicitly)."""
    sources = {
        f.name: make_encoder_source(tmp_path, f.name, frame_ids, dim=f.dim, video=video)
        for f in _FIELDS
    }
    return store.ingest(sources)


def test_ingest_aligns_and_composes_global_pk_AC2(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    # Real SPEC-0004 shape: per-video frame_ids ("0000"), repeated across
    # videos; video identity lives in the npy filename. Two videos, two
    # per-video ingest passes (as ingest-all does).
    store = milvus_lite_store(fields=_FIELDS)
    r1 = _ingest_video(store, tmp_path, "L25_V011", ["0000", "0001", "0002"])
    r2 = _ingest_video(store, tmp_path, "L07_V003", ["0000", "0001"])

    assert r1.n_rows == 3
    assert r2.n_rows == 2
    assert sorted(r1.fields_loaded) == ["metaclip2", "qwen3vl", "siglip2"]

    rows = store.client.query(
        "keyframes",
        filter="",
        output_fields=["pk", "frame_id", "video_id", "frame_idx"],
    )
    by_pk = {r["pk"]: r for r in rows}

    # Global PK is composed `<video_id>_<frame_id>`; per-video frame_ids that
    # collide across videos ("0000") become distinct PKs.
    assert set(by_pk) == {
        "L25_V011_0000",
        "L25_V011_0001",
        "L25_V011_0002",
        "L07_V003_0000",
        "L07_V003_0001",
    }

    # The per-video frame_id and video_id are kept as their own scalar fields.
    assert by_pk["L25_V011_0000"]["frame_id"] == "0000"
    assert by_pk["L25_V011_0000"]["video_id"] == "L25_V011"
    assert by_pk["L07_V003_0001"]["frame_id"] == "0001"
    assert by_pk["L07_V003_0001"]["video_id"] == "L07_V003"

    # frame_idx is 0-based within each video.
    assert by_pk["L25_V011_0000"]["frame_idx"] == 0
    assert by_pk["L25_V011_0001"]["frame_idx"] == 1
    assert by_pk["L25_V011_0002"]["frame_idx"] == 2
    assert by_pk["L07_V003_0000"]["frame_idx"] == 0
    assert by_pk["L07_V003_0001"]["frame_idx"] == 1


def test_ingest_explicit_video_id_overrides_filename_AC2(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    # An explicit video_id wins over the npy filename stem.
    store = milvus_lite_store(fields=_FIELDS)
    sources = {
        f.name: make_encoder_source(tmp_path, f.name, ["0000", "0001"], dim=f.dim, video="onbox")
        for f in _FIELDS
    }
    store.ingest(sources, video_id="L42_V123")

    rows = store.client.query("keyframes", filter="", output_fields=["pk", "video_id"])
    by_pk = {r["pk"]: r for r in rows}
    assert set(by_pk) == {"L42_V123_0000", "L42_V123_0001"}
    assert all(r["video_id"] == "L42_V123" for r in by_pk.values())


def test_ingest_missing_declared_field_raises_AC2(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    frame_ids = ["0000", "0001"]
    store = milvus_lite_store(fields=_FIELDS)
    # Only provide two of the three declared fields -> raise before any write.
    partial = {
        "siglip2": make_encoder_source(tmp_path, "siglip2", frame_ids, dim=8, video="L01_V001"),
        "metaclip2": make_encoder_source(tmp_path, "metaclip2", frame_ids, dim=4, video="L01_V001"),
    }
    import pytest

    with pytest.raises(ValueError, match="qwen3vl"):
        store.ingest(partial)
