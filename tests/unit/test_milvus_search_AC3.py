# Proves SPEC-0006 AC3: search(field, queries, top_k) returns one list per
# query of length min(top_k, n), descending by score, rank 1-based contiguous,
# score in [-1, 1].

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aic2026.index.milvus_schema import DenseField
from aic2026.index.milvus_store import MilvusKeyframeStore

from .conftest import make_encoder_source

_FIELDS = (DenseField("siglip2", 8), DenseField("metaclip2", 4))


def _ingest_fixture(store: MilvusKeyframeStore, tmp_path, n: int) -> list[str]:
    # Per-video frame_ids; video identity from the npy filename.
    frame_ids = [f"{i:04d}" for i in range(n)]
    sources = {
        f.name: make_encoder_source(tmp_path, f.name, frame_ids, dim=f.dim, video="L25_V011")
        for f in _FIELDS
    }
    store.ingest(sources)
    return frame_ids


def test_search_shape_order_and_score_range_AC3(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    store = milvus_lite_store(fields=_FIELDS)
    n = 6
    _ingest_fixture(store, tmp_path, n)

    rng = np.random.default_rng(0)
    queries = rng.standard_normal((2, 8)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    top_k = 3
    results = store.search("siglip2", queries, top_k=top_k)

    assert len(results) == 2
    for hits in results:
        assert len(hits) == min(top_k, n)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)  # descending
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))  # 1-based contiguous
        for h in hits:
            assert -1.0 - 1e-3 <= h.score <= 1.0 + 1e-3


def test_search_caps_at_collection_size_AC3(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    store = milvus_lite_store(fields=_FIELDS)
    n = 4
    _ingest_fixture(store, tmp_path, n)
    q = np.zeros((1, 8), dtype=np.float32)
    q[0, 0] = 1.0
    results = store.search("siglip2", q, top_k=100)
    assert len(results[0]) == n  # min(top_k, n)


def test_search_single_vector_is_accepted_AC3(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    store = milvus_lite_store(fields=_FIELDS)
    _ingest_fixture(store, tmp_path, 3)
    vec = np.zeros(8, dtype=np.float32)
    vec[1] = 1.0
    results = store.search("siglip2", vec, top_k=2)  # 1-D query
    assert len(results) == 1
    assert len(results[0]) == 2
