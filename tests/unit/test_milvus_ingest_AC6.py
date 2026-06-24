# Proves SPEC-0006 AC6: an .npy whose second axis disagrees with the field's
# declared dim raises an error naming the field, the expected dim, and the
# actual dim - before any write.

from __future__ import annotations

from collections.abc import Callable

import pytest

from aic2026.index.milvus_schema import DenseField
from aic2026.index.milvus_store import MilvusKeyframeStore

from .conftest import make_encoder_source

_FIELDS = (DenseField("siglip2", 8),)


def test_dim_mismatch_raises_with_diagnostic_AC6(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    store = milvus_lite_store(fields=_FIELDS)
    frame_ids = ["0000", "0001"]
    # The declared field is 8-d, but the source npy is 16-d.
    bad = {"siglip2": make_encoder_source(tmp_path, "siglip2", frame_ids, dim=16, video="L25_V011")}

    with pytest.raises(ValueError) as excinfo:
        store.ingest(bad)

    message = str(excinfo.value)
    assert "siglip2" in message
    assert "8" in message  # expected dim
    assert "16" in message  # actual dim
