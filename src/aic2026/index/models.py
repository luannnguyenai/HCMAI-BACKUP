# Implements SPEC-0006 SS 3 (pydantic models for keyframe metadata + hits).
"""Pydantic shapes for the Milvus keyframe store.

`KeyframeMeta` is the structured row payload; `Hit` is one ranked search
result; `IngestResult` is the handle returned by an ingest pass. These mirror
the SPEC-0006 SS 3 API contract.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class KeyframeMeta(BaseModel):
    """Structured scalar payload for one keyframe row."""

    model_config = ConfigDict(extra="forbid")

    pk: str = Field(description="Global primary key, composed '<video_id>_<frame_id>'.")
    frame_id: str = Field(description="Per-video frame id (the manifest stem, e.g. '001').")
    video_id: str = Field(description="Video identity from the source npy path (e.g. 'L25_V001').")
    frame_idx: int = Field(ge=0, description="0-based row index within the video.")
    youtube_url: str | None = None
    description: str | None = None
    od_tags: list[str] = Field(
        default_factory=list,
        description="Organisers' object-detection labels (advisory).",
    )


class Hit(BaseModel):
    """One ranked ANN result for a single query."""

    model_config = ConfigDict(extra="forbid")

    pk: str = Field(description="Global primary key '<video_id>_<frame_id>'.")
    frame_id: str = Field(description="Per-video frame id (the manifest stem).")
    video_id: str
    score: float = Field(description="IP score in [-1, 1] on unit vectors.")
    rank: int = Field(ge=1, description="1-based rank within the returned list.")


class IngestResult(BaseModel):
    """Outcome of one `MilvusKeyframeStore.ingest` pass."""

    model_config = ConfigDict(extra="forbid")

    collection: str
    n_rows: int = Field(ge=0)
    fields_loaded: list[str]
