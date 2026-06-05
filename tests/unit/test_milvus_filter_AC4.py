# Proves SPEC-0006 AC4: a scalar expr (e.g. video_id == 'L25_V011') restricts
# the candidate set so only matching frames appear in the ranked list.

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aic2026.index.milvus_schema import DenseField
from aic2026.index.milvus_store import MilvusKeyframeStore

from .conftest import make_encoder_source

_FIELDS = (DenseField("siglip2", 8),)


def test_scalar_filter_restricts_results_AC4(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    # Per-video frame_ids ("0000".."0003"), one ingest pass per video.
    per_video = [f"{i:04d}" for i in range(4)]
    store = milvus_lite_store(fields=_FIELDS)
    for video in ("L25_V011", "L07_V003"):
        sources = {
            f.name: make_encoder_source(tmp_path, f.name, per_video, dim=f.dim, video=video)
            for f in _FIELDS
        }
        store.ingest(sources)

    q = np.zeros((1, 8), dtype=np.float32)
    q[0, 0] = 1.0

    filtered = store.search("siglip2", q, top_k=100, expr="video_id == 'L25_V011'")
    assert filtered[0], "expected at least one hit under the filter"
    assert {h.video_id for h in filtered[0]} == {"L25_V011"}
    assert len(filtered[0]) == 4  # only the four L25_V011 frames
    assert {h.pk for h in filtered[0]} == {f"L25_V011_{i:04d}" for i in range(4)}

    # Without the filter, both videos are candidates.
    unfiltered = store.search("siglip2", q, top_k=100)
    assert {h.video_id for h in unfiltered[0]} == {"L25_V011", "L07_V003"}
