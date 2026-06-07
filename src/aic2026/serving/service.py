# Implements SPEC-0026 SS 4 (query path, frame detail, readiness).
"""The query service: encode -> per-lane ANN -> single/RRF -> ranked frames.

`QueryService` holds the SPEC-0006 `MilvusKeyframeStore` and one `Embedder`
per online lane (the SPEC-0004 text tower, ADR-0003). It is deliberately
transport-agnostic so it is unit-testable without FastAPI: the HTTP/WS layer in
`app.py` is a thin shell over these methods.

Image towers are never loaded here (ADR-0003): only `Embedder.encode_text` is
called, exactly once per requested lane (SPEC-0026 AC9).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from aic2026.embedding.base import Embedder
from aic2026.index.milvus_store import MilvusKeyframeStore
from aic2026.serving.config import ServingConfig
from aic2026.serving.fusion import FusedItem, rrf_fuse, single_lane
from aic2026.serving.models import (
    FrameDetail,
    FusionMode,
    Lane,
    QueryRequest,
    QueryResponse,
    RankedFrame,
    ReadyStatus,
)

logger = logging.getLogger(__name__)

# Scalar fields read for a frame-detail point lookup (SPEC-0006 schema).
_DETAIL_FIELDS: tuple[str, ...] = (
    "pk",
    "video_id",
    "frame_id",
    "frame_idx",
    "youtube_url",
    "description",
    "od_tags",
)


class QueryValidationError(ValueError):
    """A semantically-invalid query (maps to HTTP 422 / a WS error frame).

    Distinct from pydantic's shape validation: e.g. `fusion=rrf` with a single
    lane, or a requested lane with no configured encoder.
    """


def _thumb_url(video_id: str, frame_id: str) -> str:
    """Static thumbnail URL for a keyframe (ADR-0015 key scheme)."""
    return f"/thumbs/{video_id}/{frame_id}.jpg"


def _full_url(video_id: str, frame_id: str) -> str:
    """Static full-image URL for a keyframe (ADR-0015 key scheme)."""
    return f"/frames/{video_id}/{frame_id}.jpg"


def _to_ranked(items: list[FusedItem]) -> list[RankedFrame]:
    """Adapt fused items to wire `RankedFrame`s with contiguous 1-based ranks."""
    out: list[RankedFrame] = []
    for rank, it in enumerate(items, start=1):
        out.append(
            RankedFrame(
                pk=it.pk,
                video_id=it.video_id,
                frame_id=it.frame_id,
                rank=rank,
                score=it.score,
                thumb_url=_thumb_url(it.video_id, it.frame_id),
                full_url=_full_url(it.video_id, it.frame_id),
                per_lane={Lane(k): v for k, v in it.per_lane.items()},
            )
        )
    return out


class QueryService:
    """Wraps the SPEC-0006 store + per-lane text towers for KIS retrieval."""

    def __init__(
        self,
        store: MilvusKeyframeStore,
        encoders: dict[Lane, Embedder],
        config: ServingConfig,
    ) -> None:
        if not encoders:
            raise ValueError("at least one lane encoder is required")
        self.store = store
        self.encoders = encoders
        self.config = config

    # --- query -------------------------------------------------------------

    def query(self, req: QueryRequest) -> QueryResponse:
        """Encode once per lane, ANN per lane, single/RRF combine (SS 4)."""
        if not req.query_vi.strip():
            # Defensive: pydantic min_length=1 allows a single space.
            raise QueryValidationError("query_vi must contain a non-whitespace character")
        lanes = list(dict.fromkeys(req.lanes))  # de-dup, preserve order
        if not lanes:
            raise QueryValidationError("at least one lane is required")
        if req.fusion is FusionMode.rrf and len(lanes) < 2:
            raise QueryValidationError("fusion=rrf requires at least two lanes")
        if req.fusion is FusionMode.single and len(lanes) != 1:
            raise QueryValidationError("fusion=single requires exactly one lane")
        for lane in lanes:
            if lane not in self.encoders:
                raise QueryValidationError(f"no online encoder configured for lane {lane.value!r}")

        start = time.perf_counter()
        per_lane_hits = {}
        for lane in lanes:
            # Encode exactly once per lane (AC9): one (1, dim) text vector.
            vec = np.asarray(self.encoders[lane].encode_text([req.query_vi]), dtype=np.float32)
            hits = self.store.search(lane.value, vec, top_k=req.top_k)
            per_lane_hits[lane.value] = hits[0] if hits else []

        if req.fusion is FusionMode.rrf:
            fused = rrf_fuse(per_lane_hits, rrf_k=req.rrf_k, top_k=req.top_k)
        else:
            (only_lane,) = lanes
            fused = single_lane(only_lane.value, per_lane_hits[only_lane.value])

        took_ms = (time.perf_counter() - start) * 1000.0
        return QueryResponse(
            query_vi=req.query_vi,
            lanes=lanes,
            fusion=req.fusion,
            results=_to_ranked(fused),
            took_ms=took_ms,
        )

    # --- frame detail ------------------------------------------------------

    def frame_detail(self, pk: str) -> FrameDetail | None:
        """Point-lookup the scalar payload + same-video neighbours for `pk`."""
        client = self.store.client
        client.load_collection(self.store.collection)  # type: ignore[attr-defined]
        rows = client.get(  # type: ignore[attr-defined]
            self.store.collection,
            ids=[pk],
            output_fields=list(_DETAIL_FIELDS),
        )
        if not rows:
            return None
        row = rows[0]
        video_id = str(row["video_id"])
        frame_idx = int(row["frame_idx"])
        return FrameDetail(
            pk=str(row.get("pk", pk)),
            video_id=video_id,
            frame_id=str(row["frame_id"]),
            frame_idx=frame_idx,
            youtube_url=str(row["youtube_url"]) or None,
            description=str(row["description"]) or None,
            od_tags=[str(t) for t in (row.get("od_tags") or [])],
            ocr_text=None,  # SPEC-0005 not landed
            asr_text=None,  # SPEC-0005 not landed
            full_url=_full_url(video_id, str(row["frame_id"])),
            neighbours=self._neighbours(video_id, frame_idx),
        )

    def _neighbours(self, video_id: str, frame_idx: int) -> list[str]:
        """Prev/next pks (by frame_idx) within the same video."""
        safe_video = video_id.replace('"', '\\"')
        expr = (
            f'video_id == "{safe_video}" '
            f"and (frame_idx == {frame_idx - 1} or frame_idx == {frame_idx + 1})"
        )
        rows = self.store.client.query(  # type: ignore[attr-defined]
            self.store.collection,
            filter=expr,
            output_fields=["pk", "frame_idx"],
        )
        ordered = sorted(rows, key=lambda r: int(r["frame_idx"]))
        return [str(r["pk"]) for r in ordered]

    # --- readiness ---------------------------------------------------------

    def readiness(self) -> ReadyStatus:
        """Snapshot Milvus + thumbnail-tier health (SS 4 startup contract)."""
        collection_loaded = False
        row_count = 0
        try:
            client = self.store.client
            if client.has_collection(self.store.collection):  # type: ignore[attr-defined]
                client.load_collection(self.store.collection)  # type: ignore[attr-defined]
                collection_loaded = True
                row_count = self._row_count()
        except Exception:  # readiness must never raise; report not-ready instead
            logger.warning("readiness: Milvus probe failed", exc_info=True)

        thumbs_present = _dir_nonempty(self.config.thumb_root)
        store_fields = {f.name for f in self.store.fields}
        lanes_available = [lane for lane in self.encoders if lane.value in store_fields]
        ready = collection_loaded and row_count > 0 and thumbs_present
        return ReadyStatus(
            ready=ready,
            collection_loaded=collection_loaded,
            row_count=row_count,
            thumbnails_present=thumbs_present,
            lanes_available=lanes_available,
        )

    def _row_count(self) -> int:
        """Best-effort entity count (stats first, count(*) query as fallback)."""
        client = self.store.client
        try:
            stats = client.get_collection_stats(self.store.collection)  # type: ignore[attr-defined]
            count = int(stats.get("row_count", 0))
            if count > 0:
                return count
        except Exception:  # fall back to a count(*) query
            logger.debug("get_collection_stats failed; using count(*)", exc_info=True)
        try:
            rows = client.query(  # type: ignore[attr-defined]
                self.store.collection,
                filter="",
                output_fields=["count(*)"],
            )
            if rows:
                return int(rows[0].get("count(*)", 0))
        except Exception:  # report 0 (not-ready) rather than raise
            logger.debug("count(*) query failed", exc_info=True)
        return 0


def _dir_nonempty(root: Path) -> bool:
    """True when `root` is a directory containing at least one entry."""
    try:
        return root.is_dir() and any(root.iterdir())
    except OSError:
        return False


def build_default_service(config: ServingConfig) -> QueryService:
    """Construct the production service: real store + real text towers.

    Lazy-imports the encoders (the `embedding` extra: torch + open_clip) so this
    module imports cleanly in CI, which never installs that extra. Dev/CI tests
    inject a `QueryService` built with `DummyEmbedder` instead of calling this.
    """
    from aic2026.cli.embed import _resolve_encoder
    from aic2026.index.milvus_schema import FLOOR_FIELDS

    dim_by_name = {f.name: f.dim for f in FLOOR_FIELDS}
    dim_by_name.update(config.encoder_dim_overrides)

    store = MilvusKeyframeStore(uri=config.milvus_uri, collection=config.collection)
    encoders: dict[Lane, Embedder] = {}
    for name in config.online_lanes:
        try:
            lane = Lane(name)
        except ValueError:
            logger.warning("skipping non-online lane %r in online_lanes", name)
            continue
        encoders[lane] = _resolve_encoder(name, dim_by_name[name])
    if not encoders:
        raise ValueError(f"no online lanes resolved from {config.online_lanes!r}")
    return QueryService(store, encoders, config)
