# Shared fixtures for SPEC-0006 Milvus store tests.
"""Milvus Lite test scaffolding.

The unit tests for SPEC-0006 run against a *real* embedded Milvus Lite
instance in a tmp dir (FLAT index, exact, CPU, no network) so they exercise
the actual pymilvus surface without a GPU or a running server. Milvus Lite
spawns a local gRPC server on loopback; some sandboxed/locked-down
environments forbid that bind. When the instance cannot initialise, the
`milvus_lite_store` factory skips the test rather than failing - CI (no
sandbox) runs the real path.

Fixtures and helpers build their per-field `.npy` + manifest inputs through
the SPEC-0004 `extract_image_embeddings` path with `DummyEmbedder`, so the
ingest tests consume exactly the artifact shape the producer writes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from aic2026.embedding.dummy import DummyEmbedder
from aic2026.embedding.extract import extract_image_embeddings
from aic2026.index.milvus_schema import DenseField
from aic2026.index.milvus_store import EncoderSource, MilvusKeyframeStore


def make_encoder_source(
    root: Path,
    field_name: str,
    frame_ids: Sequence[str],
    *,
    dim: int,
    video: str = "L25_V011",
) -> EncoderSource:
    """Write a SPEC-0004 `.npy` + manifest pair for one field/video.

    This mirrors the real SPEC-0004 producer layout: the manifest `frame_id`
    is a PER-VIDEO stem (e.g. "0000"), and the video identity lives only in the
    `.npy` filename (`<video>.npy`), NOT inside `frame_id`. The store derives
    `video_id` from that filename (or an explicit arg) and composes the global
    primary key `<video_id>_<frame_id>` (SPEC-0006 SS 4).

    Fake image files named `<frame_id>.jpg` are encoded with a per-field
    `DummyEmbedder` (distinct `model_id` so lanes differ), via the real
    extraction path. Returns the `EncoderSource` the store ingests.
    """
    img_dir = root / field_name / video / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fid in frame_ids:
        p = img_dir / f"{fid}.jpg"
        p.write_bytes(f"{field_name}:{video}:{fid}".encode())
        paths.append(p)
    out = root / field_name / video
    result = extract_image_embeddings(
        paths,
        DummyEmbedder(dim=dim, model_id=f"dummy-{field_name}"),
        out=out,
        batch_size=8,
    )
    return EncoderSource(vectors=result.vectors_path, manifest=result.manifest_path)


@pytest.fixture
def milvus_lite_store(tmp_path: Path) -> Callable[..., MilvusKeyframeStore]:
    """Factory: build a Milvus Lite-backed store, skipping if Lite won't start."""
    pytest.importorskip("pymilvus", reason="pymilvus not installed (index extra)")
    pytest.importorskip("milvus_lite", reason="milvus-lite backend not installed")
    from pymilvus.exceptions import MilvusException

    # Errors Milvus Lite raises when it cannot stand up its loopback server
    # (e.g. a sandbox that forbids binding 127.0.0.1).
    init_errors: tuple[type[BaseException], ...] = (RuntimeError, OSError, MilvusException)
    counter = {"n": 0}

    def _build(
        *,
        fields: Sequence[DenseField],
        collection: str = "keyframes",
    ) -> MilvusKeyframeStore:
        counter["n"] += 1
        uri = tmp_path / f"lite_{counter['n']}.db"
        store = MilvusKeyframeStore(uri=str(uri), collection=collection, fields=fields)
        try:
            store.ensure_collection()
        except init_errors as exc:  # pragma: no cover - env-dependent skip
            pytest.skip(f"Milvus Lite could not initialise in this environment: {exc!r}")
        return store

    return _build
