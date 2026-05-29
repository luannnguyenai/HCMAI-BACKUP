# Proves SPEC-0004 AC1: encode_text / encode_image return float32 (n, dim)
# arrays with row L2 norms ~1.0, matching the encoder's declared dim.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.embedding import DummyEmbedder, l2_normalize


def test_encode_text_shape_dtype_dim_AC1() -> None:
    emb = DummyEmbedder(dim=64)
    out = emb.encode_text(["alpha", "beta", "gamma"])
    assert out.shape == (3, 64)
    assert out.dtype == np.float32
    assert out.shape[1] == emb.dim


def test_encode_text_l2_normalised_AC1() -> None:
    emb = DummyEmbedder(dim=128)
    out = emb.encode_text(["a", "bb", "ccc", "dddd"])
    norms = np.linalg.norm(out, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-3), norms


def test_encode_image_shape_and_norm_AC1(tmp_path: Path) -> None:
    files = []
    for i in range(5):
        p = tmp_path / f"img_{i:03d}.jpg"
        p.write_bytes(f"fake-bytes-{i}".encode())
        files.append(p)
    emb = DummyEmbedder(dim=32)
    out = emb.encode_image(files)
    assert out.shape == (5, 32)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-3)


def test_encode_text_empty_input_AC1() -> None:
    emb = DummyEmbedder(dim=16)
    out = emb.encode_text([])
    assert out.shape == (0, 16)
    assert out.dtype == np.float32


def test_dim_must_be_positive_AC1() -> None:
    with pytest.raises(ValueError):
        DummyEmbedder(dim=0)


def test_l2_normalize_rejects_non_2d_AC1() -> None:
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(8, dtype=np.float32))
