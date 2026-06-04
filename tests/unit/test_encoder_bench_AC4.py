# Proves SPEC-0025 AC4: measure_deployability returns a DeployStat with positive
# latency, matching dim, and CPU-path VRAM=None / fits=None; topk_indices and
# sample_keyframes behave. CPU-only, no torch.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.embedding.dummy import DummyEmbedder
from aic2026.eval.encoder_bench import (
    measure_deployability,
    sample_keyframes,
    topk_indices,
)


def test_measure_deployability_cpu_path_AC4() -> None:
    enc = DummyEmbedder(dim=64)
    stat = measure_deployability(enc, ["xin chao", "ha noi"], repeats=5)
    assert stat.model_id == enc.model_id
    assert stat.dim == 64
    assert stat.latency_p50_ms > 0.0
    assert stat.latency_p95_ms >= stat.latency_p50_ms
    # CPU/CI: no CUDA -> VRAM + fit verdict are None (SPEC-0025 SS 4).
    assert stat.vram_mb is None
    assert stat.fits_5070_headroom is None
    assert stat.quant == "cpu"
    # round-trips to a json-able dict.
    assert stat.as_dict()["dim"] == 64


def test_topk_indices_ordering_AC4() -> None:
    # doc 2 is identical to the query -> rank 1; doc 0 opposite -> last.
    docs = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    top = topk_indices(q, docs, k=2)
    assert top.shape == (1, 2)
    assert top[0, 0] == 2  # best match
    assert 2 not in top[0, 1:] or top[0, 0] == 2


def test_topk_indices_empty_AC4() -> None:
    out = topk_indices(np.zeros((2, 4), np.float32), np.zeros((0, 4), np.float32), k=5)
    assert out.shape == (2, 0)


def test_sample_keyframes_deterministic_AC4(tmp_path: Path) -> None:
    for i in range(10):
        vid = tmp_path / f"L25_V{i:03d}"
        vid.mkdir(parents=True, exist_ok=True)
        (vid / "001.jpg").write_bytes(b"x")
    a = sample_keyframes(tmp_path, 4, seed=1)
    b = sample_keyframes(tmp_path, 4, seed=1)
    assert a == b
    assert len(a) == 4
    # n >= total returns all
    allp = sample_keyframes(tmp_path, 999, seed=1)
    assert len(allp) == 10


def test_sample_keyframes_empty_raises_AC4(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \.jpg"):
        sample_keyframes(tmp_path, 4)
