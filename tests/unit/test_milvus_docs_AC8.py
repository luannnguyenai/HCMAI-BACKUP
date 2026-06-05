# Proves SPEC-0006 AC8 (by inspection, no Milvus): the store's module + class
# docstrings document the offline-ingest / online-query split (ADR-0003) and
# the qwen3vl offline-only-doc-lane asymmetry (ADR-0012), and the DenseField
# carries the `online_query` flag (False for the qwen3vl lane).

from __future__ import annotations

from aic2026.index import milvus_store
from aic2026.index.milvus_schema import FLOOR_FIELDS, DenseField


def test_module_docstring_documents_offline_online_split_AC8() -> None:
    doc = milvus_store.__doc__ or ""
    assert "ADR-0003" in doc  # offline ingest / online query split
    assert "ADR-0012" in doc  # qwen3vl offline-only visual-document lane
    assert "offline" in doc.lower()
    assert "online" in doc.lower()


def test_dense_field_flags_qwen3vl_offline_lane_AC8() -> None:
    by_name = {f.name: f for f in FLOOR_FIELDS}
    assert by_name["qwen3vl"].online_query is False
    assert by_name["siglip2"].online_query is True
    assert by_name["metaclip2"].online_query is True
    # The flag is documentation, not a guard: qwen3vl is still a real,
    # queryable dense field in this component (SS 9 Q-d).
    assert "online_query" in (DenseField.__doc__ or "")
