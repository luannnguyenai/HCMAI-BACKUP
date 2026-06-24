# Implements SPEC-0006 SS 3 (collection schema + index params).
"""Declarative schema for the multi-vector keyframe collection.

One Milvus collection (`keyframes`) holds several named dense vector fields -
one per image-text encoder - keyed by a global frame primary key, alongside
the structured scalar fields from research-note 06 SS 2.3. All vectors are
L2-normalised by the SPEC-0004 producer contract, so the metric is inner
product (IP), which on unit vectors equals cosine.

This module is pure data: no `pymilvus` import. The store
(`aic2026.index.milvus_store`) turns these declarations into a live schema.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseField:
    """One named dense vector field in the keyframe collection.

    `online_query=False` marks the qwen3vl offline visual-document lane
    (ADR-0012): the field is fully searchable here (offline-derived query
    vectors), but there is no online text encoder producing a qwen3vl query
    vector on the finals 5070 hot path. This flag is documentation, not a
    runtime guard - the offline/online split is enforced by deployment.
    """

    name: str  # "siglip2", "metaclip2", "qwen3vl"
    dim: int  # 1152 / 1024 / 2048
    metric: str = "IP"  # IP on unit vectors == cosine
    online_query: bool = True  # False for qwen3vl (offline doc lane, ADR-0012)


# The shipped floor (SS 9 Q-b RESOLVED: defer qwen8b 4096 and provided CLIP
# 512; re-ingest from R2 is cheap if they earn their place under SPEC-0015).
FLOOR_FIELDS: tuple[DenseField, ...] = (
    DenseField("siglip2", 1152),
    DenseField("metaclip2", 1024),
    DenseField("qwen3vl", 2048, online_query=False),
)


@dataclass(frozen=True)
class HnswParams:
    """HNSW index params, adopted verbatim from proposal 01 SS 5.4.

    Only used on the standalone/managed deployment (the real ingest target,
    SS 9 Q-a). Milvus Lite (dev/CI) supports FLAT only, so these are ignored
    there; the store selects FLAT when the uri is a local Lite path.
    """

    M: int = 32
    ef_construction: int = 200
    # Raised from the proposal-01 default of 128 to the SS 11.8-validated
    # recall-passing operating point: on stable 2.5.x, ef=1024 is the lowest
    # swept point that clears recall@200 >= 0.95 on all three lanes (siglip2
    # 0.969, metaclip2 0.968, qwen3vl 0.985), with p95 latency 16.6-51.1 ms,
    # well under the 150 ms NFR. The server also requires ef >= top_k; `search`
    # additionally clamps ef = max(ef, top_k) so any caller default stays valid.
    ef_search: int = 1024


# The global primary key field. Its value is composed `<video_id>_<frame_id>`
# by the store at ingest time (SPEC-0006 SS 4): `video_id` comes from the
# source npy path, `frame_id` is the per-video manifest stem. The PK is NOT
# parsed back out of any single scalar; both parts are kept as their own
# scalar fields below so structured filtering on `video_id` / `frame_id` works.
PRIMARY_KEY: str = "pk"

# Scalar (structured) fields carried per frame, in the order they are added to
# the schema. research-note 06 SS 2.3. `video_id` is the video identity (from
# the source npy path); `frame_id` is the per-video manifest stem. Neither is
# the primary key (that is `PRIMARY_KEY` above).
SCALAR_FIELDS: tuple[str, ...] = (
    "video_id",
    "frame_idx",
    "frame_id",
    "youtube_url",
    "description",
    "od_tags",
)

# VARCHAR / ARRAY capacities. Surfaced as named constants per AGENTS.md
# ("no magic numbers"); sized for the AIC frame-id / organiser-metadata shapes
# (research-note 06 SS 2.3). Generous floors, not tuned limits.
PK_MAX_LEN: int = 384  # `<video_id>_<frame_id>`; >= VIDEO_ID_MAX_LEN + 1 + FRAME_ID_MAX_LEN.
FRAME_ID_MAX_LEN: int = 256
VIDEO_ID_MAX_LEN: int = 64
YOUTUBE_URL_MAX_LEN: int = 512
DESCRIPTION_MAX_LEN: int = 8192
OD_TAG_MAX_LEN: int = 128
OD_TAGS_MAX_CAPACITY: int = 128
