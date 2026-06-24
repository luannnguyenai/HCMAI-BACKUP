# Implements SPEC-0006 SS 3 (public surface of the Milvus keyframe store).
"""Multi-vector Milvus keyframe store (SPEC-0006).

Schema + index params live in `milvus_schema`, pydantic shapes in `models`,
and the live store + query path + Submission adapter in `milvus_store`.
"""

from aic2026.index.milvus_schema import (
    FLOOR_FIELDS,
    DenseField,
    HnswParams,
)
from aic2026.index.milvus_store import (
    DEFAULT_COLLECTION,
    DEFAULT_TOP_K,
    EncoderSource,
    MilvusBackend,
    MilvusKeyframeStore,
    hits_to_submissions,
)
from aic2026.index.models import Hit, IngestResult, KeyframeMeta

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_TOP_K",
    "FLOOR_FIELDS",
    "DenseField",
    "EncoderSource",
    "Hit",
    "HnswParams",
    "IngestResult",
    "KeyframeMeta",
    "MilvusBackend",
    "MilvusKeyframeStore",
    "hits_to_submissions",
]
