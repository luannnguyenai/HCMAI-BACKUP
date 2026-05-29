# Proves SPEC-0004 AC3: extract_image_embeddings writes .npy + a manifest
# whose rows align with the matrix; re-runs are byte-identical with the
# deterministic dummy encoder.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aic2026.embedding import DummyEmbedder
from aic2026.embedding.extract import (
    L2_NORM_TOLERANCE,
    ExtractionResult,
    discover_images,
    extract_image_embeddings,
)


def _seed_dir(tmp_path: Path, n: int) -> Path:
    """Create `n` fake image files; returns the directory path."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(n):
        (d / f"frame_{i:03d}.jpg").write_bytes(f"fake-bytes-{i}".encode())
    # A non-image sibling that must be filtered out.
    (d / "notes.txt").write_text("ignore me", encoding="utf-8")
    return d


def test_extract_writes_aligned_npy_and_manifest_AC3(tmp_path: Path) -> None:
    in_dir = _seed_dir(tmp_path, n=7)
    out = tmp_path / "out" / "vectors"
    paths = discover_images(in_dir)
    assert len(paths) == 7  # the .txt was excluded

    result = extract_image_embeddings(paths, DummyEmbedder(dim=32), out=out, batch_size=3)

    assert isinstance(result, ExtractionResult)
    assert result.n == 7
    assert result.dim == 32

    matrix = np.load(result.vectors_path)
    assert matrix.shape == (7, 32)
    assert matrix.dtype == np.float32
    norms = np.linalg.norm(matrix, axis=1)
    assert np.all(np.abs(norms - 1.0) < L2_NORM_TOLERANCE)

    with result.manifest_path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 7
    for i, row in enumerate(rows):
        assert row["row"] == i
        assert row["frame_id"] == paths[i].stem
        assert row["path"] == str(paths[i])


def test_extract_is_deterministic_byte_identical_AC3(tmp_path: Path) -> None:
    in_dir = _seed_dir(tmp_path, n=5)
    paths = discover_images(in_dir)

    out_a = tmp_path / "a" / "v"
    out_b = tmp_path / "b" / "v"
    extract_image_embeddings(paths, DummyEmbedder(dim=24), out=out_a, batch_size=2)
    extract_image_embeddings(paths, DummyEmbedder(dim=24), out=out_b, batch_size=2)

    assert (out_a.parent / "v.npy").read_bytes() == (out_b.parent / "v.npy").read_bytes()
    assert (out_a.parent / "v.manifest.jsonl").read_text(encoding="utf-8") == (
        out_b.parent / "v.manifest.jsonl"
    ).read_text(encoding="utf-8")


def test_extract_empty_input_writes_zero_rows_AC3(tmp_path: Path) -> None:
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    paths = discover_images(in_dir)
    out = tmp_path / "out" / "v"
    result = extract_image_embeddings(paths, DummyEmbedder(dim=16), out=out, batch_size=4)
    assert result.n == 0
    matrix = np.load(result.vectors_path)
    assert matrix.shape == (0, 16)
    assert result.manifest_path.read_text(encoding="utf-8") == ""


def test_extract_rejects_zero_batch_size_AC3(tmp_path: Path) -> None:
    in_dir = _seed_dir(tmp_path, n=2)
    paths = discover_images(in_dir)
    with pytest.raises(ValueError):
        extract_image_embeddings(paths, DummyEmbedder(dim=8), out=tmp_path / "v", batch_size=0)
