# Implements SPEC-0026 SS 3 (API request/response models).
"""Pydantic request/response shapes for the MVP serving API.

These mirror the SPEC-0026 SS 3 contract one-to-one and are the single source
of truth the SPEC-0027 TypeScript types track. `extra="forbid"` everywhere so a
client typo is a 422, not a silently-dropped field.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Lane(StrEnum):
    """An online queryable encoder lane (ADR-0003 text-tower lanes).

    `qwen3vl` is deliberately absent: it is the offline-only visual-document
    lane (ADR-0012) with no online text encoder, so it is not selectable here.
    """

    siglip2 = "siglip2"
    metaclip2 = "metaclip2"


class FusionMode(StrEnum):
    """How multiple lanes are combined into one ranked list."""

    single = "single"  # one lane, scores passed through
    rrf = "rrf"  # reciprocal-rank fusion over >= 2 lanes (ADR-0008)


class QueryRequest(BaseModel):
    """One KIS free-text query (SPEC-0026 SS 3)."""

    model_config = ConfigDict(extra="forbid")

    query_vi: str = Field(min_length=1, max_length=512)
    lanes: list[Lane] = Field(default_factory=lambda: [Lane.siglip2])
    top_k: int = Field(default=48, ge=1, le=500)
    fusion: FusionMode = FusionMode.single
    rrf_k: int = Field(default=60, ge=1)  # honoured only when fusion == rrf


class RankedFrame(BaseModel):
    """One ranked keyframe in a `QueryResponse`."""

    model_config = ConfigDict(extra="forbid")

    pk: str  # SPEC-0006 global pk "<video_id>_<frame_id>"
    video_id: str
    frame_id: str
    rank: int = Field(ge=1)  # 1-based, post-fusion
    score: float  # fused score (IP for single, RRF score for rrf)
    thumb_url: str  # static thumbnail path
    full_url: str  # static full-image path
    per_lane: dict[Lane, float] = Field(default_factory=dict)  # pre-fusion lane scores


class QueryResponse(BaseModel):
    """The ranked result for one query."""

    model_config = ConfigDict(extra="forbid")

    query_vi: str
    lanes: list[Lane]
    fusion: FusionMode
    results: list[RankedFrame]
    took_ms: float  # server-side wall-clock for encode + search + fuse


class FrameDetail(BaseModel):
    """Scalar metadata + neighbours for one keyframe (SPEC-0026 SS 4)."""

    model_config = ConfigDict(extra="forbid")

    pk: str
    video_id: str
    frame_id: str
    frame_idx: int
    youtube_url: str | None = None
    description: str | None = None
    od_tags: list[str] = Field(default_factory=list)
    ocr_text: str | None = None  # present only when SPEC-0005 lands; else None
    asr_text: str | None = None  # present only when SPEC-0005 lands; else None
    full_url: str
    neighbours: list[str] = Field(default_factory=list)  # prev/next pks in same video


class IssueReport(BaseModel):
    """A tester-filed reproducible issue (SPEC-0026 SS 4 / SPEC-0027)."""

    model_config = ConfigDict(extra="forbid")

    query_vi: str
    lanes: list[Lane]
    fusion: FusionMode
    returned_frame_ids: list[str]  # pks shown when the report was filed
    screenshot_png_b64: str  # base64 PNG of the UI at report time
    client_timestamp: str  # ISO-8601 from the browser
    note: str | None = Field(default=None, max_length=2000)


class IssueResponse(BaseModel):
    """Where the issue landed (GitHub URL or local fallback path)."""

    model_config = ConfigDict(extra="forbid")

    issue_url: str | None  # GitHub issue URL, or None on local-fallback
    fallback_path: str | None  # local path when GitHub is unavailable


class ReadyStatus(BaseModel):
    """Readiness snapshot for `/readyz` (SPEC-0026 SS 4)."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    collection_loaded: bool
    row_count: int
    thumbnails_present: bool
    lanes_available: list[Lane]
