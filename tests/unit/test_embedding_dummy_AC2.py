# Proves SPEC-0004 AC2: DummyEmbedder is deterministic per (model_id, dim,
# input); distinct inputs yield distinct vectors.

from __future__ import annotations

from pathlib import Path

import numpy as np

from aic2026.embedding import DummyEmbedder


def test_encode_text_is_deterministic_AC2() -> None:
    a = DummyEmbedder(dim=64).encode_text(["xin chao", "hello", "bonjour"])
    b = DummyEmbedder(dim=64).encode_text(["xin chao", "hello", "bonjour"])
    np.testing.assert_array_equal(a, b)


def test_encode_image_is_deterministic_AC2(tmp_path: Path) -> None:
    files = []
    for i in range(4):
        p = tmp_path / f"f_{i}.jpg"
        p.write_bytes(f"content-{i}".encode())
        files.append(p)
    a = DummyEmbedder(dim=48).encode_image(files)
    b = DummyEmbedder(dim=48).encode_image(files)
    np.testing.assert_array_equal(a, b)


def test_distinct_text_inputs_distinct_vectors_AC2() -> None:
    out = DummyEmbedder(dim=32).encode_text(["foo", "bar", "baz", "qux"])
    # No two rows should be identical (collision-free on this tiny set).
    for i in range(out.shape[0]):
        for j in range(i + 1, out.shape[0]):
            assert not np.array_equal(out[i], out[j]), (i, j)


def test_distinct_image_bytes_distinct_vectors_AC2(tmp_path: Path) -> None:
    files = []
    for i in range(3):
        p = tmp_path / f"x_{i}.png"
        p.write_bytes(f"payload-{i}".encode())
        files.append(p)
    out = DummyEmbedder(dim=32).encode_image(files)
    assert not np.array_equal(out[0], out[1])
    assert not np.array_equal(out[1], out[2])


def test_model_id_changes_vectors_AC2() -> None:
    """Different model_id with same input must produce different vectors."""
    inp = ["the same query"]
    a = DummyEmbedder(dim=16, model_id="dummy-a").encode_text(inp)
    b = DummyEmbedder(dim=16, model_id="dummy-b").encode_text(inp)
    assert not np.array_equal(a, b)
