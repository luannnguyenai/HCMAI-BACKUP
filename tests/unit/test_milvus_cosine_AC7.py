# Proves SPEC-0006 AC7: querying with a stored document's own vector returns
# that document at rank 1 with score within 1.0 +- 1e-3 (IP on unit vectors ==
# cosine). Under Milvus Lite FLAT the search is exact, so recall@k vs brute
# force is 1.0 by construction; HNSW recall@200 >= 0.95 (SS 6) is a lease-box
# integration check, not exercisable on Lite (FLAT-only).

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aic2026.index.milvus_schema import DenseField
from aic2026.index.milvus_store import MilvusKeyframeStore

from .conftest import make_encoder_source

_FIELDS = (DenseField("siglip2", 8),)
_COSINE_TOL = 1e-3  # SPEC-0006 SS 4 cosine-identity tolerance.


def test_self_query_returns_rank1_score_one_AC7(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    n = 8
    frame_ids = [f"{i:04d}" for i in range(n)]  # per-video stems
    source = make_encoder_source(tmp_path, "siglip2", frame_ids, dim=8, video="L25_V011")
    store = milvus_lite_store(fields=_FIELDS)
    store.ingest({"siglip2": source})

    stored = np.load(source.vectors)  # (n, 8), unit-norm by SPEC-0004 contract

    # Query with each stored vector; it must come back at rank 1, score ~1.0.
    for k in (0, 3, n - 1):
        results = store.search("siglip2", stored[k], top_k=5)
        hits = results[0]
        assert hits[0].frame_id == frame_ids[k]
        assert hits[0].pk == f"L25_V011_{frame_ids[k]}"
        assert abs(hits[0].score - 1.0) <= _COSINE_TOL
        # Exact (FLAT) ranking: the self-hit dominates every other score.
        assert all(hits[0].score >= other.score for other in hits[1:])


def test_flat_ranking_is_exact_full_recall_AC7(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    n = 12
    frame_ids = [f"{i:04d}" for i in range(n)]  # per-video stems
    source = make_encoder_source(tmp_path, "siglip2", frame_ids, dim=8, video="L25_V011")
    store = milvus_lite_store(fields=_FIELDS)
    store.ingest({"siglip2": source})
    stored = np.load(source.vectors)

    q = stored[2]
    brute = np.argsort(-(stored @ q))[:5]  # exact top-5 by cosine
    hits = store.search("siglip2", q, top_k=5)[0]
    got = [frame_ids.index(h.frame_id) for h in hits]
    assert got == list(brute)  # FLAT == brute force, recall 1.0
