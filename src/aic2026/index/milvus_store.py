# Implements SPEC-0006 SS 3-4 (offline ingest + online ANN query).
"""The multi-vector keyframe store and its query path.

`MilvusKeyframeStore` owns one Milvus collection with several named dense
vector fields (the SPEC-0004 encoder floor) keyed by a global primary key
`pk` composed `<video_id>_<frame_id>`. It has:

- `ensure_collection`  idempotent schema + per-field ANN index creation.
- `ingest`             read SPEC-0004 `<enc>.npy` + `<enc>.manifest.jsonl`
                       pairs for one video, align by the per-video `frame_id`,
                       upsert one entity per frame under a global `pk`.
- `search`             per-field top-k ANN with an optional scalar filter.

Primary-key contract (SPEC-0006 SS 4, reconciled with SPEC-0004): the
SPEC-0004 manifest `frame_id` is a PER-VIDEO stem (e.g. "001"); the video
identity lives in the source npy filename (e.g. `L25_V001.npy`), NOT inside
`frame_id`. So `video_id` is derived from the source (the `.npy` path or an
explicit `video_id` arg), and the global `pk` is composed `<video_id>_<frame_id>`.
The per-video `frame_id` and the `video_id` are kept as their own scalar
fields so structured filtering still works.

`hits_to_submissions` adapts one ranked list to SPEC-0001 `Submission` rows,
and `MilvusBackend` wraps an encoder + the store as a single-lane
`harness.Backend` (multi-lane fusion + task-type routing is SPEC-0015).

Offline / online split (ADR-0003): ingestion is offline (GH200-class lease);
the query path receives an already-encoded vector and never loads an encoder.
The `qwen3vl` field is the offline-only visual-document lane (ADR-0012): it is
fully searchable here by offline-derived query vectors - there is simply no
online text encoder that produces a qwen3vl query vector on the finals 5070
hot path. There is deliberately no runtime guard preventing qwen3vl queries
(SS 9 Q-d RESOLVED).

Deployment (SS 9 Q-a RESOLVED): dev/CI run against the embedded, file-backed
Milvus Lite mode (CPU, no network, FLAT index); the real HNSW build runs on
Milvus standalone (docker) on the lease box. `pymilvus` is lazy-imported so
this module imports cleanly on a box without the `index` extra.

Engine pin (SPEC-0006 SS 12): the milvus-lite 3.0 engine cross-wires a
multi-vector collection's per-field segments when a fresh process reopens a
persisted `.db` (the ingest -> serve pattern), so `pymilvus`/`milvus-lite` are
pinned `< 3` in the `index` extra. The pin is the fix; this module's schema /
ingest / search logic is engine-agnostic and unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aic2026.index.milvus_schema import (
    DESCRIPTION_MAX_LEN,
    FLOOR_FIELDS,
    FRAME_ID_MAX_LEN,
    OD_TAG_MAX_LEN,
    OD_TAGS_MAX_CAPACITY,
    PK_MAX_LEN,
    PRIMARY_KEY,
    VIDEO_ID_MAX_LEN,
    YOUTUBE_URL_MAX_LEN,
    DenseField,
    HnswParams,
)
from aic2026.index.models import Hit, IngestResult

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from aic2026.embedding.base import Embedder
    from aic2026.models.submission import Submission
    from aic2026.models.task import MockTask

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION: str = "keyframes"
DEFAULT_TOP_K: int = 200  # SPEC-0006 SS 3 search default.
DEFAULT_BATCH_SIZE: int = 1000  # SPEC-0006 SS 3 ingest default.

# Lite (FLAT) vs standalone (HNSW). Milvus Lite supports FLAT only; the real
# HNSW build runs on standalone (SS 7, SS 9 Q-a).
_LITE_INDEX_TYPE: str = "FLAT"
_STANDALONE_INDEX_TYPE: str = "HNSW"

# Network endpoint schemes; anything else is treated as a local Milvus Lite
# file path.
_REMOTE_URI_PREFIXES: tuple[str, ...] = ("http://", "https://", "tcp://", "grpc://", "unix:")

# Separator joining `<video_id>` and `<frame_id>` into the global primary key.
_PK_SEP: str = "_"


@dataclass(frozen=True)
class EncoderSource:
    """A SPEC-0004 output pair for one encoder/video."""

    vectors: Path  # <enc>.npy, float32 (n, dim)
    manifest: Path  # <enc>.manifest.jsonl, rows {row, frame_id, path}


def _is_lite_uri(uri: str) -> bool:
    """True when `uri` is a local Milvus Lite file path (not a network endpoint)."""
    return not uri.startswith(_REMOTE_URI_PREFIXES)


def _compose_pk(video_id: str, frame_id: str) -> str:
    """Global primary key from a video identity and a per-video frame id."""
    return f"{video_id}{_PK_SEP}{frame_id}"


def _read_manifest_frame_ids(manifest: Path) -> list[str]:
    """Return frame_ids in manifest row order (SPEC-0004 `{row, frame_id, path}`)."""
    frame_ids: list[str] = []
    with Path(manifest).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_ids.append(str(row["frame_id"]))
    return frame_ids


def _read_metadata(metadata: Path) -> dict[str, dict[str, object]]:
    """Read an optional scalar jsonl keyed by the global `pk`.

    Each line is a JSON object carrying the global `pk` (`<video_id>_<frame_id>`)
    plus any of `youtube_url`, `description`, `od_tags`. A row may instead carry
    a `frame_id` (per-video) and `video_id`, from which the `pk` is composed.
    Missing keys default at insert time.
    """
    out: dict[str, dict[str, object]] = {}
    with Path(metadata).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "pk" in row:
                key = str(row["pk"])
            elif "video_id" in row and "frame_id" in row:
                key = _compose_pk(str(row["video_id"]), str(row["frame_id"]))
            else:
                key = str(row["frame_id"])
            out[key] = row
    return out


class MilvusKeyframeStore:
    """Owns one multi-vector collection.

    `uri` is a Milvus Lite file path (dev/CI) or a standalone/managed endpoint
    (real ingest); see SPEC-0006 SS 7. The index type is selected from the uri:
    FLAT for Lite, HNSW for a network endpoint.
    """

    def __init__(
        self,
        *,
        uri: str,
        collection: str = DEFAULT_COLLECTION,
        fields: Sequence[DenseField] = FLOOR_FIELDS,
        index: HnswParams | None = None,
    ) -> None:
        if not fields:
            raise ValueError("at least one DenseField is required")
        self.uri = uri
        self.collection = collection
        self.fields: tuple[DenseField, ...] = tuple(fields)
        self.index = index or HnswParams()
        self.is_lite = _is_lite_uri(uri)
        self._client: object | None = None
        self._by_name: dict[str, DenseField] = {f.name: f for f in self.fields}

    # --- client ------------------------------------------------------------

    @property
    def client(self) -> object:
        """Lazily-constructed `pymilvus.MilvusClient` (heavy import deferred)."""
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:  # pragma: no cover - exercised manually
                raise ImportError(
                    "pymilvus is required for the Milvus store; install the extra "
                    "with `uv sync --extra index` (SPEC-0006 SS 7)."
                ) from exc
            self._client = MilvusClient(uri=self.uri)
        return self._client

    def _field(self, name: str) -> DenseField:
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(self._by_name)
            raise ValueError(f"unknown dense field {name!r}; declared: {known}") from None

    # --- schema ------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Idempotent. Creates the collection + per-field ANN index if absent."""
        from pymilvus import DataType

        client = self.client
        if client.has_collection(self.collection):  # type: ignore[attr-defined]
            return

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)  # type: ignore[attr-defined]
        # Global primary key `<video_id>_<frame_id>` (SPEC-0006 SS 4). The
        # per-video `frame_id` and the `video_id` are kept as their own scalar
        # fields so structured filtering on either still works.
        schema.add_field(PRIMARY_KEY, DataType.VARCHAR, is_primary=True, max_length=PK_MAX_LEN)
        schema.add_field("video_id", DataType.VARCHAR, max_length=VIDEO_ID_MAX_LEN)
        schema.add_field("frame_idx", DataType.INT64)
        schema.add_field("frame_id", DataType.VARCHAR, max_length=FRAME_ID_MAX_LEN)
        schema.add_field("youtube_url", DataType.VARCHAR, max_length=YOUTUBE_URL_MAX_LEN)
        schema.add_field("description", DataType.VARCHAR, max_length=DESCRIPTION_MAX_LEN)
        schema.add_field(
            "od_tags",
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=OD_TAGS_MAX_CAPACITY,
            max_length=OD_TAG_MAX_LEN,
        )
        for f in self.fields:
            schema.add_field(f.name, DataType.FLOAT_VECTOR, dim=f.dim)

        index_params = client.prepare_index_params()  # type: ignore[attr-defined]
        index_type = _LITE_INDEX_TYPE if self.is_lite else _STANDALONE_INDEX_TYPE
        for f in self.fields:
            params: dict[str, int] = (
                {}
                if self.is_lite
                else {"M": self.index.M, "efConstruction": self.index.ef_construction}
            )
            index_params.add_index(
                field_name=f.name,
                index_type=index_type,
                metric_type=f.metric,
                params=params,
            )
        client.create_collection(self.collection, schema=schema, index_params=index_params)  # type: ignore[attr-defined]
        logger.info(
            "created collection %r (%s index) with fields %s",
            self.collection,
            index_type,
            [f.name for f in self.fields],
        )

    # --- ingest ------------------------------------------------------------

    def ingest(
        self,
        field_sources: Mapping[str, EncoderSource],
        *,
        video_id: str | None = None,
        metadata: Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> IngestResult:
        """Ingest one video: align field sources by the per-video `frame_id`,
        compose the global `pk`, build full entities, upsert.

        `video_id` is the video identity. When `None` it is derived from the
        first declared field's `.npy` filename stem (SPEC-0004 / ADR-0011
        layout: `<index_root>/<enc>/<video>.npy`). The global primary key is
        composed `<video_id>_<frame_id>`; `video_id` is NOT parsed back out of
        any single scalar.

        All dense fields in `self.fields` must be present in `field_sources`
        (Milvus has no nullable vector fields; upsert overwrites all fields, so
        a partial-field upsert is impossible). Raises before any write when a
        declared field is missing, a dim disagrees, or a manifest/vector length
        disagrees.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        self.ensure_collection()

        missing = [f.name for f in self.fields if f.name not in field_sources]
        if missing:
            declared = [f.name for f in self.fields]
            raise ValueError(
                f"field_sources is missing declared dense field(s) {missing}; "
                f"all of {declared} must be present (no nullable vector fields)."
            )
        unknown = [name for name in field_sources if name not in self._by_name]
        if unknown:
            raise ValueError(
                f"field_sources has undeclared field(s) {unknown}; declared: {list(self._by_name)}"
            )

        # Load + validate each field's matrix and manifest.
        loaded: dict[str, tuple[list[str], np.ndarray]] = {}
        for name, src in field_sources.items():
            field = self._by_name[name]
            matrix = np.load(src.vectors)
            actual_dim = matrix.shape[1] if matrix.ndim == 2 else None
            if matrix.ndim != 2 or actual_dim != field.dim:
                raise ValueError(
                    f"dim mismatch for field {name!r}: expected dim {field.dim}, "
                    f"actual {actual_dim} (npy shape {tuple(matrix.shape)})"
                )
            frame_ids = _read_manifest_frame_ids(src.manifest)
            if len(frame_ids) != matrix.shape[0]:
                raise ValueError(
                    f"manifest/vector length mismatch for field {name!r}: "
                    f"{len(frame_ids)} manifest rows vs {matrix.shape[0]} vectors"
                )
            loaded[name] = (frame_ids, matrix.astype(np.float32, copy=False))

        # Canonical frame order = the first declared field's manifest order.
        canonical_name = self.fields[0].name
        canonical_ids = loaded[canonical_name][0]
        canonical_set = set(canonical_ids)

        # video_id is derived from the SOURCE, not parsed from frame_id: an
        # explicit arg if given, else the first field's npy filename stem
        # (`<index_root>/<enc>/<video>.npy`). One ingest call = one video.
        resolved_video_id = video_id or field_sources[canonical_name].vectors.stem
        if not resolved_video_id:
            raise ValueError(
                "could not resolve video_id: pass video_id explicitly or name the "
                f"source npy `<video>.npy` (got stem "
                f"{field_sources[canonical_name].vectors.stem!r})"
            )
        id_to_row = {
            name: {fid: i for i, fid in enumerate(fids)} for name, (fids, _) in loaded.items()
        }
        for name, (fids, _) in loaded.items():
            if set(fids) != canonical_set:
                raise ValueError(
                    f"frame_id set mismatch between {canonical_name!r} and {name!r}; "
                    "all encoder lanes must cover the same frames."
                )

        meta = _read_metadata(metadata) if metadata is not None else {}

        entities: list[dict[str, object]] = []
        for frame_idx, fid in enumerate(canonical_ids):
            pk = _compose_pk(resolved_video_id, fid)
            row_meta = meta.get(pk, {})
            entity: dict[str, object] = {
                PRIMARY_KEY: pk,
                "frame_id": fid,
                "video_id": resolved_video_id,
                "frame_idx": frame_idx,
                "youtube_url": str(row_meta.get("youtube_url") or ""),
                "description": str(row_meta.get("description") or ""),
                "od_tags": [str(t) for t in (row_meta.get("od_tags") or [])],
            }
            for name, (_, matrix) in loaded.items():
                entity[name] = matrix[id_to_row[name][fid]].tolist()
            entities.append(entity)

        n_rows = len(entities)
        for start in range(0, n_rows, batch_size):
            self.client.upsert(self.collection, entities[start : start + batch_size])  # type: ignore[attr-defined]

        logger.info("ingested %d rows into %r", n_rows, self.collection)
        return IngestResult(
            collection=self.collection,
            n_rows=n_rows,
            fields_loaded=list(loaded.keys()),
        )

    # --- query -------------------------------------------------------------

    def search(
        self,
        field: str,
        queries: np.ndarray,
        *,
        top_k: int = DEFAULT_TOP_K,
        expr: str | None = None,
        ef_search: int | None = None,
    ) -> list[list[Hit]]:
        """Per-field top-k ANN. One ranked list per query row, descending score.

        `queries` is `(nq, dim)` (a 1-D `(dim,)` vector is accepted as a single
        query). `expr` is an optional Milvus scalar filter applied before
        ranking. Scores are IP on unit vectors == cosine, in `[-1, 1]`.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive; got {top_k}")
        dense = self._field(field)
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2 or q.shape[1] != dense.dim:
            raise ValueError(
                f"query dim mismatch for field {field!r}: expected (nq, {dense.dim}), "
                f"got shape {tuple(q.shape)}"
            )

        search_params: dict[str, object] | None = None
        if not self.is_lite:
            # Standalone HNSW requires efSearch >= top_k (a stable 2.5.x server
            # rejects ef < k outright; SPEC-0006 SS 11.8). Clamp so any caller
            # default is always a valid operating point.
            ef = ef_search if ef_search is not None else self.index.ef_search
            ef = max(ef, top_k)
            search_params = {"params": {"ef": ef}}

        # A collection opened in a fresh process starts 'released'; ANN search
        # requires it loaded into memory first. load is idempotent and a no-op
        # when already loaded (e.g. right after create/ingest in-process).
        self.client.load_collection(self.collection)  # type: ignore[attr-defined]

        results = self.client.search(  # type: ignore[attr-defined]
            self.collection,
            data=q.tolist(),
            anns_field=field,
            limit=top_k,
            filter=expr or "",
            output_fields=[PRIMARY_KEY, "frame_id", "video_id"],
            search_params=search_params,
        )

        out: list[list[Hit]] = []
        for per_query in results:
            ranked: list[Hit] = []
            for rank, raw in enumerate(per_query, start=1):
                entity = raw.get("entity", {})
                ranked.append(
                    Hit(
                        pk=str(entity.get(PRIMARY_KEY, raw.get("id", ""))),
                        frame_id=str(entity["frame_id"]),
                        video_id=str(entity["video_id"]),
                        score=float(raw["distance"]),
                        rank=rank,
                    )
                )
            out.append(ranked)
        return out


def hits_to_submissions(hits: Sequence[Hit]) -> list[Submission]:
    """Adapt one ranked list to SPEC-0001 `Submission` rows.

    Ranks are re-emitted contiguous 1-based in the given order; `score` and
    the global `pk` are carried through as the `Submission.frame_id` (the
    answer identity scored against ground truth, which is globally unique).
    This is not a `Backend.search` impl - fusion and task-type logic are
    SPEC-0015.
    """
    from aic2026.models.submission import Submission

    return [
        Submission(rank=i, score=hit.score, frame_id=hit.pk) for i, hit in enumerate(hits, start=1)
    ]


class MilvusBackend:
    """Single-lane `harness.Backend` (SPEC-0001) backed by one dense field.

    Encodes the task's Vietnamese query with an injected `Embedder`, runs a
    single-field ANN search, and adapts the hits to `Submission` rows. This is
    the single-lane search backing; multi-lane fusion and per-task-type routing
    are SPEC-0015. It satisfies the `Backend` protocol's `search(task,
    time_budget_ms)` signature.
    """

    def __init__(
        self,
        store: MilvusKeyframeStore,
        encoder: Embedder,
        *,
        field: str = "siglip2",
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.field = field
        self.top_k = top_k

    def search(self, task: MockTask, time_budget_ms: int) -> list[Submission]:
        _ = time_budget_ms  # latency budgeting is harness-side (SPEC-0001).
        q = np.asarray(self.encoder.encode_text([task.query_vi]), dtype=np.float32)
        hits = self.store.search(self.field, q, top_k=self.top_k)
        return hits_to_submissions(hits[0] if hits else [])
