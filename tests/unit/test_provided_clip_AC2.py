# Proves SPEC-0025 AC2: ProvidedClipEmbedder.encode_image looks up pre-extracted
# vectors by composite frame id, L2-normalises, and honours strict / zero-fill on
# a missing id; from_dir parses the matrix+ids (layout A) feature store. All
# offline / no network (the CLIP text tower is importorskip-gated separately).

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aic2026.embedding.provided_clip import ProvidedClipEmbedder


def _mapping(dim: int = 512) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "L25_V001_001": rng.standard_normal(dim).astype(np.float32),
        "L25_V001_002": rng.standard_normal(dim).astype(np.float32),
        "L26_V010_050": rng.standard_normal(dim).astype(np.float32),
    }


def test_encode_image_lookup_by_composite_id_AC2() -> None:
    emb = ProvidedClipEmbedder(_mapping())
    # default key_fn: "<parentDir>_<stem>" -> matches "L25_V001_001"
    out = emb.encode_image([Path("/x/L25_V001/001.jpg"), Path("/x/L26_V010/050.jpg")])
    assert out.shape == (2, 512)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_encode_image_strict_missing_raises_AC2() -> None:
    emb = ProvidedClipEmbedder(_mapping(), strict=True)
    with pytest.raises(KeyError, match="no provided-CLIP feature"):
        emb.encode_image([Path("/x/L99_V999/777.jpg")])


def test_encode_image_nonstrict_zero_fills_AC2() -> None:
    emb = ProvidedClipEmbedder(_mapping(), strict=False)
    out = emb.encode_image([Path("/x/L99_V999/777.jpg")])
    assert out.shape == (1, 512)
    assert np.allclose(out[0], 0.0)


def test_from_dir_layout_a_matrix_plus_ids_AC2(tmp_path: Path) -> None:
    mat = np.random.default_rng(1).standard_normal((3, 512)).astype(np.float32)
    np.save(tmp_path / "feats.npy", mat)
    (tmp_path / "ids.json").write_text(
        json.dumps(["L25_V001_001", "L25_V001_002", "L26_V010_050"]), encoding="utf-8"
    )
    emb = ProvidedClipEmbedder.from_dir(tmp_path)
    assert emb.dim == 512
    out = emb.encode_image([Path("/x/L25_V001/002.jpg")])
    assert out.shape == (1, 512)
    assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-3


def test_from_dir_unparseable_raises_AC2(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("not features", encoding="utf-8")
    with pytest.raises(ValueError, match="could not parse provided-CLIP"):
        ProvidedClipEmbedder.from_dir(tmp_path)


def test_empty_features_rejected_AC2() -> None:
    with pytest.raises(ValueError, match="features mapping is empty"):
        ProvidedClipEmbedder({})
