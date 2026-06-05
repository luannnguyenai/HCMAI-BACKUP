# Proves SPEC-0006 AC1: ensure_collection creates one collection with each
# DenseField as a named vector field of the correct dim + IP metric, plus the
# six scalar fields, and re-running is a no-op. Runs against a real Milvus Lite
# instance (FLAT) in a tmp dir; skips if Lite cannot initialise here.

from __future__ import annotations

from collections.abc import Callable

from aic2026.index.milvus_schema import FLOOR_FIELDS, PRIMARY_KEY, SCALAR_FIELDS
from aic2026.index.milvus_store import MilvusKeyframeStore


def test_ensure_collection_creates_fields_and_metric_AC1(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
) -> None:
    store = milvus_lite_store(fields=FLOOR_FIELDS, collection="keyframes")
    client = store.client

    assert client.has_collection("keyframes")
    desc = client.describe_collection("keyframes")
    fields_by_name = {f["name"]: f for f in desc["fields"]}

    # The global primary key `<video_id>_<frame_id>` is its own field, distinct
    # from the per-video frame_id scalar (SPEC-0006 SS 4).
    assert PRIMARY_KEY in fields_by_name, f"missing primary key field {PRIMARY_KEY!r}"
    assert fields_by_name[PRIMARY_KEY].get("is_primary") is True
    assert PRIMARY_KEY not in SCALAR_FIELDS

    # All six structured scalar fields are present.
    for scalar in SCALAR_FIELDS:
        assert scalar in fields_by_name, f"missing scalar field {scalar!r}"

    # Each dense field exists at its declared dim with the IP metric.
    for dense in FLOOR_FIELDS:
        assert dense.name in fields_by_name, f"missing vector field {dense.name!r}"
        assert int(fields_by_name[dense.name]["params"]["dim"]) == dense.dim
        index = client.describe_index("keyframes", dense.name)
        assert index["metric_type"] == "IP"


def test_ensure_collection_is_idempotent_AC1(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
) -> None:
    store = milvus_lite_store(fields=FLOOR_FIELDS, collection="keyframes")
    desc_before = store.client.describe_collection("keyframes")
    # Re-running must be a no-op: same collection, same field set.
    store.ensure_collection()
    desc_after = store.client.describe_collection("keyframes")
    assert {f["name"] for f in desc_before["fields"]} == {f["name"] for f in desc_after["fields"]}
    assert store.client.list_collections() == ["keyframes"]
