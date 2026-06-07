# Implements SPEC-0026 SS 5 AC9 (encode once per lane; never load an image tower).
"""AC9: the query path encodes each query once per lane (no duplicate encode)
and never loads an image tower (ADR-0003).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aic2026.embedding.dummy import DummyEmbedder
from aic2026.serving.models import FusionMode, Lane, QueryRequest
from aic2026.serving.service import QueryService


class CountingEmbedder:
    """Wraps a DummyEmbedder, counting text encodes and forbidding image ones."""

    def __init__(self, *, dim: int, model_id: str) -> None:
        self._inner = DummyEmbedder(dim=dim, model_id=model_id)
        self.dim = dim
        self.model_id = model_id
        self.text_calls = 0
        self.image_calls = 0

    def encode_text(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        return self._inner.encode_text(texts)

    def encode_image(self, paths: list[Path]) -> np.ndarray:  # pragma: no cover - must not run
        self.image_calls += 1
        raise AssertionError("the online query path must never load an image tower (ADR-0003)")


def test_each_lane_encoded_exactly_once_AC9(serving_env) -> None:
    siglip = CountingEmbedder(dim=8, model_id="c-siglip2")
    metaclip = CountingEmbedder(dim=8, model_id="c-metaclip2")
    service = QueryService(
        serving_env.store,
        {Lane.siglip2: siglip, Lane.metaclip2: metaclip},
        serving_env.config,
    )

    service.query(
        QueryRequest(
            query_vi="canh hoang hon tren bien",
            lanes=[Lane.siglip2, Lane.metaclip2],
            fusion=FusionMode.rrf,
            top_k=5,
        )
    )

    assert siglip.text_calls == 1
    assert metaclip.text_calls == 1
    assert siglip.image_calls == 0
    assert metaclip.image_calls == 0
