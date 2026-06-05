---
id: SPEC-0006
title: Milvus schema and queries (multi-vector keyframe store + structured filter)
status: Implementing
owner: unassigned
created: 2026-06-05
updated: 2026-06-05
implements_proposal: docs/proposals/01-interactive-system-architecture.md SS 5.4
related_adrs:
  - ADR-0003
  - ADR-0011
  - ADR-0012
depends_on:
  - SPEC-0001
  - SPEC-0004
  - SPEC-0025
---

# SPEC-0006 - Milvus schema and queries

> The storage and ANN-query layer for the dense image-text retrieval floor. It defines one Milvus collection that holds multiple named dense vector fields (one per encoder) keyed by a global keyframe identity, an offline ingestion path that loads the `.npy` + manifest pairs that SPEC-0004 produces, and an online query path that returns per-encoder top-k ranked lists with optional structured filtering. It replaces the brute-force `DenseRetriever` (SPEC-0025) with ANN. It does not fuse the lanes (SPEC-0015) and does not encode queries (SPEC-0004).

## 1. Context

[`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) SS 5.4 names Milvus 2.5+ as the vector store with an HNSW index (M=32, efConstruction=200, efSearch=128). SS 2.3 of that proposal sketches the storage as one collection per encoder. This spec supersedes that sketch with a **single multi-vector collection** (SS 2 below), reconciled in SS 9; Milvus 2.5 multi-vector fields did not exist when the proposal was written.

[ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md) splits the workload: image-tower vectors are extracted and pre-indexed **offline** on the GH200-class lease box; **only text-tower query encoding runs online** on the 12 GB RTX 5070. This spec's **ingestion path is offline**; its **query path is online** (it receives an already-encoded query vector and runs ANN; it does not load an encoder).

The data this targets is real, not synthetic. Per [ADR-0011](../adr/ADR-0011-r2-artifact-store-and-lease-rollover.md) / SPEC-0022 the banked indexes live in Cloudflare R2:

- `index/aic2025-proxy-3enc-20260604/<enc>/<video>.npy` + `<video>.manifest.jsonl` for `enc in {siglip2, metaclip2, qwen3vl}`, **546 videos** each.
- `index/aic2025-proxy-qwen8b-20260604/<video>.npy` (+ manifest).

Per-encoder dims (verified on the lease): siglip2 **1152**, metaclip2 **1024**, qwen3vl **2048** (offline-only visual-document lane per [ADR-0012](../adr/ADR-0012-qwen-offline-visual-document-lane.md)), qwen8b **4096**, organisers' provided CLIP **512** (research-note 06 SS 2.4 / 07). All vectors are L2-normalised by the SPEC-0004 contract, so the metric is **inner product (IP) = cosine**. The SPEC-0004 manifest `frame_id` is a **per-video** stem (e.g. `"001"`); the video identity lives in the `.npy` filename (`<video>.npy`, e.g. `L25_V001.npy`), **not** inside `frame_id`. This store therefore derives `video_id` from the source path and composes the **global primary key** as `<video_id>_<frame_id>` (SS 4); video grouping is `L<NN>_V<NNN>`.

## 2. Scope

### 2.1 In scope
- A Milvus **collection schema**: one collection, multiple named dense vector fields (siglip2 1152, metaclip2 1024, qwen3vl 2048) keyed by a global primary key `pk` (composed `<video_id>_<frame_id>`), plus the structured/scalar fields from research-note 06 SS 2.3 (`video_id`, `frame_idx`, `frame_id`, `youtube_url`, `description`, `od_tags`). The per-video `frame_id` and the `video_id` are kept as their own scalar fields (for filtering); neither is the primary key.
- An **offline ingestion path**: read the SPEC-0004 `<enc>.npy` + `<enc>.manifest.jsonl` pairs (from R2-mirrored local paths) one video at a time, align lanes by the per-video `frame_id`, derive `video_id` from the source, compose the global `pk`, and upsert.
- An **online query path**: given query vector(s) for one named field, return per-field top-k ranked lists, with an optional scalar filter expression.
- A thin adapter that maps one ranked list to `list[Submission]` (SPEC-0001) so the query path can back a future `Backend.search`.

### 2.2 Out of scope
- **C2 multi-lane fusion** (SPEC-0015). This spec returns per-field ranked lists; it does not combine them.
- **The online query encoder** (SPEC-0004 / ADR-0003). The query path receives a vector; it never loads a model.
- **GT-based accuracy eval** (R@k / nDCG). No ground truth exists yet (SPEC-0025 SS 9 Q1); accuracy is gated until it lands.
- **The qwen8b 4096-d lane and the organisers' 512-d provided CLIP** as live fields (open questions Q-b).
- **Elasticsearch / OCR / ASR / caption** indexes (SPEC-0007).
- **Production HA / sharding / replication** of the Milvus deployment.

## 3. API contract / interface

```python
# aic2026/index/milvus_schema.py

from dataclasses import dataclass

@dataclass(frozen=True)
class DenseField:
    name: str               # "siglip2", "metaclip2", "qwen3vl"
    dim: int                # 1152 / 1024 / 2048
    metric: str = "IP"      # IP on unit vectors == cosine
    online_query: bool = True   # False for qwen3vl: offline doc lane, no online encoder (ADR-0012)

# The shipped floor. qwen8b (4096) and clip_organiser (512) are deferred (SS 9 Q-b).
FLOOR_FIELDS: tuple[DenseField, ...] = (
    DenseField("siglip2", 1152),
    DenseField("metaclip2", 1024),
    DenseField("qwen3vl", 2048, online_query=False),
)

@dataclass(frozen=True)
class HnswParams:
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 1024  # SS 11.8-validated recall-passing point; `search` clamps ef = max(ef, top_k)
```

```python
# aic2026/index/models.py

from pydantic import BaseModel, ConfigDict

class KeyframeMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pk: str                       # global primary key, composed "<video_id>_<frame_id>"
    frame_id: str                 # per-video frame id (manifest stem, e.g. "001")
    video_id: str                 # "L<NN>_V<NNN>" from the source npy path
    frame_idx: int                # 0-based row index within the video
    youtube_url: str | None = None
    description: str | None = None
    od_tags: list[str] = []       # organisers' object-detection labels (advisory)

class Hit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pk: str                       # global primary key "<video_id>_<frame_id>"
    frame_id: str                 # per-video frame id (manifest stem)
    video_id: str
    score: float                  # IP score in [-1, 1]
    rank: int                     # 1-based

class IngestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: str
    n_rows: int
    fields_loaded: list[str]
```

```python
# aic2026/index/milvus_store.py

from collections.abc import Mapping, Sequence
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class EncoderSource:
    vectors: Path        # <enc>.npy, float32 (n, dim)
    manifest: Path       # <enc>.manifest.jsonl, rows {row, frame_id, path}

class MilvusKeyframeStore:
    """Owns one multi-vector collection. `uri` is a Milvus Lite file path
    (dev/CI) or a standalone/managed endpoint (real ingest); see SS 7."""

    def __init__(
        self,
        *,
        uri: str,
        collection: str = "keyframes",
        fields: Sequence[DenseField] = FLOOR_FIELDS,
        index: HnswParams | None = None,
    ) -> None: ...

    def ensure_collection(self) -> None:
        """Idempotent. Creates the collection + per-field ANN index if absent."""
        ...

    def ingest(
        self,
        field_sources: Mapping[str, EncoderSource],
        *,
        video_id: str | None = None,    # source video identity; default = npy filename stem
        metadata: Path | None = None,   # optional scalar jsonl keyed by pk
        batch_size: int = 1000,
    ) -> IngestResult:
        """Ingest one video: align field sources by the per-video frame_id,
        derive video_id from the source (the explicit arg, else the first
        field's <video>.npy filename stem), compose the global primary key
        pk = "<video_id>_<frame_id>", build full entities, upsert.
        All dense fields in `fields` must be present in `field_sources`
        (Milvus has no nullable vector fields; upsert overwrites all fields)."""
        ...

    def search(
        self,
        field: str,
        queries: np.ndarray,            # (nq, dim), L2-normalised
        *,
        top_k: int = 200,
        expr: str | None = None,        # scalar filter, e.g. "video_id == 'L25_V011'"
        ef_search: int | None = None,
    ) -> list[list[Hit]]: ...

def hits_to_submissions(hits: Sequence[Hit]) -> list["Submission"]:
    """Adapt one ranked list to SPEC-0001 Submission rows. The global pk is
    carried as Submission.frame_id (the answer identity scored against ground
    truth, which is globally unique); ranks are re-emitted contiguous 1-based.
    Not a Backend.search impl; fusion + task-type logic is SPEC-0015."""
    ...

class MilvusBackend:
    """Single-lane harness.Backend (SPEC-0001): encode the task's query with an
    injected Embedder, ANN one dense field, adapt hits to Submissions. Multi-lane
    fusion + per-task-type routing is SPEC-0015; this is the single-lane backing."""

    def __init__(
        self,
        store: MilvusKeyframeStore,
        encoder: "Embedder",
        *,
        field: str = "siglip2",
        top_k: int = 200,
    ) -> None: ...

    def search(self, task: "MockTask", time_budget_ms: int) -> list["Submission"]: ...
```

`search` loads the collection before querying (a collection opened in a fresh
process starts in the `released` state; `load_collection` is idempotent).

```
bin/index ingest  --uri URI --collection keyframes --index-root DIR --video L25_V011
                  [--fields siglip2,metaclip2,qwen3vl] [--metadata meta.jsonl]
bin/index ingest-all --uri URI --collection keyframes --index-root DIR
                  [--fields ...] [--metadata meta.jsonl]
bin/index search  --uri URI --collection keyframes --field siglip2
                  (--query-npy q.npy | --query-text "..." [--encoder dummy --dim N])
                  [--top-k 200] [--expr "video_id == 'L25_V011'"]
```

`--index-root` points at an R2-mirrored local tree (`<enc>/<video>.npy` + manifest); the CLI derives per-field paths by the SPEC-0004 / ADR-0011 convention. `search` takes exactly one of `--query-npy` (a pre-encoded `(nq, dim)` / `(dim,)` float32 array) or `--query-text` (encoded online via the injected `--encoder`; only `dummy` is CI-safe, real encoders need the `embedding` extra).

## 4. Behaviour

- **Normal ingest**: one video per call. For each field, load `(n, dim)` float32 and the byte-aligned manifest; assert `dim` equals the field's declared `dim` and the manifest row count equals `n`. Align fields across encoders by the **per-video** `frame_id` (the manifest stem). Derive `video_id` from the **source** - an explicit `video_id` arg if given, else the first field's `<video>.npy` filename stem - **not** by parsing `frame_id`. Compose the global primary key `pk = "<video_id>_<frame_id>"`; `frame_idx` is the 0-based row within the video. Build one entity per frame carrying `pk`, the per-video `frame_id`, `video_id`, all dense fields + scalars, upsert in `batch_size` chunks.
- **Empty input**: `ensure_collection` runs; `ingest` over zero rows returns `n_rows=0` and exits 0.
- **Missing field**: a field declared in `fields` but absent from `field_sources` raises before any write (no partial entities; Milvus upsert overwrites all fields, so partial-field upsert is not possible).
- **Dim mismatch**: an `.npy` whose second axis differs from the field's declared `dim` raises a clear error naming field, expected dim, actual dim.
- **Manifest/vector length mismatch**: raises; never silently truncates.
- **Normal search**: returns one ranked list per query row, each length `min(top_k, collection_size)`, ordered by descending IP score, `rank` 1-based, `score` in `[-1, 1]`.
- **Filtered search**: `expr` restricts candidates to matching scalars before ANN ranking; only matching frames appear.
- **Cosine identity**: querying with a document's own stored vector returns that document at rank 1 with `score` within `1.0 +- 1e-3` (IP on unit vectors).
- **Offline/online split**: ingestion is offline (ADR-0003); search receives a pre-encoded vector and never loads an encoder. The `qwen3vl` field is `online_query=False`: it is a doc-vector-only lane (ADR-0012), so the online path has no encoder that produces a `qwen3vl` query vector (see SS 9 Q-d).

## 5. Acceptance criteria

- **AC1**: `ensure_collection` creates a collection with each `DenseField` as a named vector field of the correct `dim` and IP metric, plus the scalar fields `video_id`, `frame_idx`, `frame_id`, `youtube_url`, `description`, `od_tags`; re-running is a no-op. Verified in `tests/unit/test_milvus_schema_AC1.py` (Milvus Lite).
- **AC2**: `ingest` aligns `.npy` rows to manifest rows by the per-video `frame_id`, derives `video_id` from the source (the `<video>.npy` filename stem or an explicit arg, **not** parsed from `frame_id`), composes the global primary key `pk = "<video_id>_<frame_id>"` (so per-video frame ids that repeat across videos become distinct PKs), keeps the per-video `frame_id` and `video_id` as their own scalar fields, sets `frame_idx` 0-based within the video, and `IngestResult.n_rows` equals the row count. Verified in `tests/unit/test_milvus_ingest_AC2.py`.
- **AC3**: `search(field, queries, top_k)` returns one list per query of length `min(top_k, n)`, descending by score, `rank` 1-based, `score` in `[-1, 1]`. Verified in `tests/unit/test_milvus_search_AC3.py`.
- **AC4**: a scalar `expr` (e.g. `video_id == 'L25_V011'`) returns only frames matching the filter. Verified in `tests/unit/test_milvus_filter_AC4.py`.
- **AC5**: `hits_to_submissions` maps a ranked list to `list[Submission]` (SPEC-0001) with matching `rank`/`score`, the global `pk` carried as `Submission.frame_id` (the answer identity scored against globally-unique ground truth), and contiguous 1-based ranks. Verified in `tests/unit/test_milvus_submission_adapter_AC5.py`.
- **AC6**: an `.npy` whose dim disagrees with the field's declared dim raises an error naming field + expected + actual dim. Verified in `tests/unit/test_milvus_ingest_AC6.py`.
- **AC7**: querying with a stored document's own vector returns that document at rank 1 with `score` within `1.0 +- 1e-3` (IP-on-unit-vectors == cosine). Verified in `tests/unit/test_milvus_cosine_AC7.py`.
- **AC8**: spec + module docstrings document the offline-ingest / online-query split (ADR-0003) and the `qwen3vl` offline-only-doc-lane asymmetry (ADR-0012). Verified by inspection, encoded as a lightweight assertion in `tests/unit/test_milvus_docs_AC8.py` (no Milvus needed).

## 6. Non-functional requirements

- **Query latency**: single-field ANN search at `top_k=200`, HNSW `efSearch=1024` (the SS 11.8-validated recall-passing point; `search` clamps `ef = max(ef, top_k)`), p95 < 150 ms over ~1M vectors on the standalone deployment (consistent with proposal 01 SS 4 "Milvus ANN per collection p50 80 ms / p95 150 ms"). On Milvus Lite dev fixtures (<= 10k rows, FLAT) p95 < 50 ms.
- **Ingestion throughput**: the full proxy single-collection build (3 fields x 546 videos) completes in < 30 min on the lease box via batched insert.
- **Recall**: HNSW recall@200 vs exact FLAT >= 0.95 at `efSearch=1024` (validated on the proxy, SS 11.8), measured on a fixture against brute force.
- **Memory**: per proposal 01 SS 5.4, ~2 GB per 1M x 1024 fp16 field on disk; the online Milvus client buffer fits the ~2 GB headroom on the 5070 (ADR-0003). The offline standalone build runs on the GH200-class lease, not the 5070.
- **Compatibility**: Python 3.11+; `pymilvus >= 2.5`; NumPy >= 1.26.

## 7. Dependencies

- **Internal**: SPEC-0004 (producer of `<enc>.npy` + `<enc>.manifest.jsonl`); SPEC-0001 (`Submission`); SPEC-0025 (`DenseRetriever`, the brute-force precedent this replaces with ANN).
- **External**: `pymilvus[milvus_lite] >= 2.5` (resolves to pymilvus 3.0 + milvus-lite 3.0 as of 2026-06-05), exposed two ways: a runtime `[project.optional-dependencies] index` extra (`uv sync --extra index`) **and** the `dev` dependency group, so CI installs it and runs the AC1-AC7 tests against a real embedded instance. Note: pymilvus 3.0 unbundled the embedded backend into the `milvus_lite` sub-extra, so it must be requested explicitly (`pymilvus[milvus_lite]`); a plain `pymilvus` install raises `milvus-lite is required for local database connections`. **Dev/CI uses Milvus Lite** (the embedded, file-backed mode: CPU-only, no network, FLAT index; it spawns a loopback gRPC server, so a sandbox that forbids binding `127.0.0.1` will skip the tests - CI does not). **The real ingest uses Milvus standalone (docker)** on the lease box (Q-a RESOLVED) - Lite's index/metric support is a subset, so CI exercises FLAT while production uses HNSW. The store selects the index type from the `uri`: FLAT for a local Lite path, HNSW for an `http(s)://` endpoint.
- **Data**: the R2-banked indexes (ADR-0011): `index/aic2025-proxy-3enc-20260604/<enc>/<video>.{npy,manifest.jsonl}` (siglip2/metaclip2/qwen3vl, 546 videos) and `index/aic2025-proxy-qwen8b-20260604/` (qwen8b). Organiser `youtube_url` / `description` / OD tags land with the June-25 AIC2026 corpus; for the proxy these scalar fields are sparse/absent and `metadata` is optional.

## 8. Test plan

- **Unit tests** (`tests/unit/`, Milvus Lite, CPU/offline/no-network; vectors from `DummyEmbedder` written via SPEC-0004 `extract`):
  - `test_milvus_schema_AC1.py` - collection + field dims + idempotent ensure.
  - `test_milvus_ingest_AC2.py` - npy/manifest alignment, frame_id/video_id parsing, row count.
  - `test_milvus_search_AC3.py` - ranked-list shape, ordering, score range.
  - `test_milvus_filter_AC4.py` - scalar filter restricts results.
  - `test_milvus_submission_adapter_AC5.py` - Hit -> Submission mapping.
  - `test_milvus_ingest_AC6.py` - dim-mismatch error.
  - `test_milvus_cosine_AC7.py` - self-query rank-1 score ~1.0; HNSW-vs-FLAT recall on a fixture.
- **Integration (lease, not CI)**: `bin/index ingest-all` over the real 546-video proxy on Milvus standalone; record ingest wall-clock + a query-latency sample in SS 10.

## 9. Open questions

All four resolved 2026-06-05 (human-directed) before implementation; recorded here.

- **Q-a RESOLVED**: Milvus Lite for dev/CI + Milvus standalone (docker) on the lease box for the real HNSW build. Rationale: Lite gives a real CPU/offline embedded instance the unit tests run against (FLAT, exact); the standalone box is the only target that builds the HNSW index proposal 01 SS 5.4 specifies. The store picks the index from the uri (FLAT for a Lite path, HNSW for an endpoint); standalone ingest is documented here, not built in this slice.
- **Q-b RESOLVED**: Defer the qwen8b (4096) and organisers' provided-CLIP (512) lanes. Ship exactly three dense fields now - siglip2 (1152), metaclip2 (1024), qwen3vl (2048). Rationale: re-ingest from the R2-banked `.npy` is cheap, so adding a lane later costs an extraction pass, not a redesign; defer until C2 fusion (SPEC-0015) shows the extra lanes earn their place against ground truth.
- **Q-c RESOLVED**: Single multi-vector collection (`keyframes`). Rationale: one frame identity and one scalar filter outweigh the coupling cost (Milvus has no nullable vector fields, so all declared lanes must be present at insert; `ingest` enforces this and raises before any write). Per-encoder collections were rejected for duplicating scalars and forcing cross-collection joins by `frame_id`.
- **Q-d RESOLVED**: `qwen3vl` is fully searchable in this component. Rationale: its query text can be encoded offline on the GH200/lease, so the field is a real, queryable dense lane here; there is deliberately **no runtime guard** preventing qwen3vl queries. The ADR-0012 offline-only constraint applies only to the finals online 5070 hot path, where qwen3vl is a fusion/re-rank signal (deferred to SPEC-0015). `online_query=False` on the field is documentation of that hot-path asymmetry, not a query block.

## 10. Reconciliation with proposal 01 SS 5.4

- **Per-encoder collections -> single multi-vector collection**: proposal SS 2.3 enumerates `collection_siglip2` / `collection_metaclip2` / etc. We use one collection with named vector fields (SS 2, Q-c). Treated as a pre-Milvus-2.5 framing the proposal could not anticipate; deviation surfaced here.
- **Field set / dims**: proposal SS 2.3 lists `internvideo2` (768) and `clap_audio` (512); neither is banked yet. The schema is built around the actually-extracted encoders: siglip2 **1152** and metaclip2 **1024** (both match the proposal as corrected in SPEC-0004), plus the **qwen3vl** 2048 offline lane (added by ADR-0012, post-dates the proposal). qwen8b 4096 and organiser CLIP 512 are deferred (Q-b). InternVideo2 / CLAP fields are reserved for when those encoders are extracted.
- **Offline-only qwen**: consistent with proposal SS 5.3 and ADR-0012; reflected as `online_query=False` on the `qwen3vl` field (Q-d).
- **Index params**: HNSW M=32 / efConstruction=200 adopted verbatim from proposal SS 5.4. The proposal's `efSearch=128` is **raised to 1024** (the SS 11.8-validated recall-passing point on stable 2.5.x); `search` also clamps `ef = max(ef, top_k)` because the 2.5.x server rejects `ef < top_k`.

## 11. Lease-box integration evidence (proxy, 2026-06-05)

First full-corpus standalone run on the H200 lease box. GT-free (no relevance
labels exist yet, SS 2.2); this validates deployment, ingest, latency and
ANN-vs-exact recall, not retrieval accuracy. Artifacts banked to R2
`eval/milvus-proxy-20260605/` (`latency_recall.json`, `qualitative.json`,
`qualitative.html`); raw logs on the box under `/tmp/aic2025/milvus_eval/`.

### 11.1 Deployment path: standalone HNSW (not Lite)

- Milvus **standalone** via the upstream `standalone_embed.sh` (single container,
  embedded etcd + local storage), image `milvusdb/milvus:v3.0-beta`, endpoint
  `http://127.0.0.1:19530`. Client `pymilvus 3.0.0` (matches the lockfile; no dep
  drift). Index type selected from the `http://` uri -> **HNSW** (M=32,
  efConstruction=200); `describe_index` confirms `state=Finished`,
  `indexed_rows=121457`, `pending_index_rows=0`.
- Docker required `sudo` (user not in the `docker` group); a foreign NIM
  container holds `:8000` (untouched). Two start-up deviations, both fixed:
  (1) host port `2379` already bound (a foreign etcd) -> removed the
  `-p 2379:2379` publish (etcd is container-internal for standalone);
  (2) the container runs as a non-root uid and could not `mkdir` the
  bind-mounted data dir -> pre-created `volumes/milvus` world-writable.
- Milvus Lite (FLAT) was the documented fallback and was **not** needed.

### 11.2 frame_id / primary-key contract mismatch (producer vs store)

The real SPEC-0004 proxy manifests set `frame_id` to the **per-video frame
number** (e.g. `"001"`), with the video identity living only in the
npy/manifest **filename** (`L25_V001`) and the `path`. But this store uses
`frame_id` as the **global primary key** and parses `video_id` from it
(`parse_video_id`, SS 4 / AC2). Ingesting the proxy as-is therefore (a) raises
in `parse_video_id` ("frame_id '001' does not contain an 'L<NN>_V<NNN>' video
id") and (b) would collide PKs across videos ("001" repeats 546 times). This
contradicts SS 1's assumption that the manifest stem already carries the video
id. The v3.0-beta run (SS 11.1-11.7) worked around it on-box by building a
sibling tree `index_milvus/<enc>/` that symlinks the unchanged `.npy` and
rewrites each manifest to `frame_id = "<video>_<frame>"` (e.g. `L25_V001_001`);
vectors untouched.

**RESOLVED 2026-06-05 (fix now in the store, no manifest rewrite):** SPEC-0006
derives `video_id` from the **source** (the `<video>.npy` filename stem or an
explicit `ingest(video_id=...)` arg) and composes the **global primary key**
`pk = "<video_id>_<frame_id>"`. The per-video `frame_id` and `video_id` are
kept as their own scalar fields. `parse_video_id` (PK-from-`frame_id`) is
removed (SS 3, SS 4, AC2). The stable 2.5.x re-run (SS 11.8) ingests the
unmodified SPEC-0004 `index/` tree directly via `bin/index ingest-all` - no
`index_milvus/` symlink tree, no on-box manifest hand-patch - validating the
fix end-to-end on real data.

### 11.3 Ingest

- `bin/index ingest-all --uri http://127.0.0.1:19530 --index-root index_milvus`
  over the full proxy: **546 videos, 121457 rows** (one entity per frame, each
  carrying all 3 dense fields), `fields=siglip2,metaclip2,qwen3vl`.
- Wall-clock **58 s** (epoch 1780631552 -> 1780631610). Exact `count(*)`
  after flush = **121457** (the pre-flush stat of 120485 was an un-flushed
  growing segment). Well under the SS 6 < 30 min NFR. **PASS.**

### 11.4 Latency (H200 proxy, not the finals 5070 number)

20 KIS text queries, per-query single-field ANN, `top_k=200`. H200 is an
indicative proxy for the 5070 NFR, not the finals figure; corpus is 121k, not
the 1M the SS 6 NFR references.

| lane | dim | p50 (ms) | p95 (ms) |
|---|---|---|---|
| siglip2 | 1152 | 12.45 | 13.42 |
| metaclip2 | 1024 | 11.04 | 11.92 |
| qwen3vl | 2048 | 11.64 | 12.80 |

All p95 far under the SS 6 < 150 ms target. **PASS** (corpus/hardware caveats
above).

### 11.5 Recall@200 (HNSW vs exact-IP numpy baseline)

20 cross-modal KIS text queries; baseline = brute-force IP over the full
in-RAM lane matrix (vectors are unit-norm, so IP == cosine).

| lane | recall_set@200 | recall_score@200 | NFR >= 0.95 |
|---|---|---|---|
| siglip2 | 0.858 | 0.862 | NOT MET |
| metaclip2 | 0.838 | 0.839 | NOT MET |
| qwen3vl | 0.927 | 0.933 | NOT MET |

(`recall_set` = id-set overlap; `recall_score` = fraction of HNSW hits scoring
>= the exact k-th score, tie-robust. Boundary tie multiplicity ~200, i.e. no
tie degeneracy, so the two agree.)

**Root cause (diagnosed, not an index defect):** the `v3.0-beta` server
**ignores the client-supplied `efSearch`**. An ef sweep {128, 256, 512}
produced byte-identical recall on every lane; a controlled probe confirms it:
- a **stored** vector retrieves the numpy-exact top-200 (**200/200**) at
  ef in {8, 128, 2048} -> the HNSW graph is exact-quality on-manifold;
- a **random off-manifold** unit vector recalls ~0.63 (124-131/200),
  ef-invariant -> the server is not honouring ef, so recall cannot be lifted
  from the client.

Cross-modal text query vectors sit off the image-vector manifold, so they land
between these extremes (0.84-0.93) and **cannot be tuned up via the client ef on
this beta**. **Verdict: the SS 6 recall@200 >= 0.95 NFR is NOT demonstrated on
`v3.0-beta` for the online cross-modal text path.** This is a beta limitation
(client ef ignored), not a defect in the schema or ingest.
**REMEDIATION:** re-run on a stable **Milvus 2.5.x** standalone (pymilvus
2.5.x) where `efSearch` is honoured, and/or raise efConstruction/M; treat the
recall NFR as **deferred** until that run. Note `efSearch=128 < top_k=200` is
itself questionable for HNSW (ef should be >= limit); the stable re-run should
set ef >= 200.

### 11.6 Qualitative

Top-5 frame_ids per lane for all 20 queries: `qualitative.html` (side-by-side
contact sheet of ids+scores) and `qualitative.json` (machine-readable), banked
to R2 `eval/milvus-proxy-20260605/` for human review.

### 11.7 NFR summary

| NFR (SS 6) | Result | Verdict |
|---|---|---|
| Ingest full proxy < 30 min | 58 s | PASS |
| Query p95 < 150 ms | 11.9-13.4 ms (121k, H200) | PASS (scale/hw caveat) |
| Recall@200 >= 0.95 | 0.84-0.93, ef untunable on beta | NOT MET / deferred |
| Compatibility (pymilvus >= 2.5) | client 3.0.0, server v3.0-beta | met |

### 11.8 Stable Milvus 2.5.x re-run (recall NFR settled, 2026-06-05)

Second full-corpus standalone run, on a **stable Milvus 2.5.27** (not the
`v3.0-beta` of 11.1-11.7), to settle the recall NFR the beta could not
demonstrate (client `efSearch` ignored, 11.5). Client **pymilvus 2.5.18**,
installed on-box for this run only - the lockfile pin of 3.0.0 is **untouched
and uncommitted**. Same H200 lease box, same 121457-row proxy corpus, same 20
KIS text queries. Artifacts banked to R2 `eval/milvus-proxy-25x-20260605/`
(`latency_recall.json`, `qualitative.json`, `qualitative.html`).

- **Deployment**: image `milvusdb/milvus:v2.5.27` via the same
  `standalone_embed.sh` with the same two deviations as 11.1 (the `-p 2379:2379`
  publish dropped; `volumes/milvus` pre-created world-writable), endpoint
  `http://127.0.0.1:19530`, HNSW (M=32, efConstruction=200), IP metric.
  `describe_index` after build: `state=Finished`, `indexed_rows=121457`,
  `pending_index_rows=0` on every lane.
- **Ingest (validates the Part 1 fix end-to-end)**: `bin/index ingest-all` over
  the **unmodified** `/tmp/aic2025/index` tree (per-video `frame_id`; **no**
  `index_milvus/` symlink tree, **no** manifest hand-patch). 546 videos /
  **121457 rows in 64 s** (epoch 1780635174 -> 1780635238); flushed
  `count(*) = 121457` (matches 11.3). A sample row confirms the composed
  contract: `pk=L26_V350_001`, `frame_id=001`, `video_id=L26_V350`.

**efSearch is honoured on 2.5.x** (the central fix vs the beta). Two proofs:
- the server **rejects** `ef < top_k`: `ef=128` at `k=200` raises
  `ef(128) should be larger than k(200)` - so the sweep is `ef in {256,512,1024}`;
- recall **moves with ef** (the beta's was byte-identical across {128,256,512}).

Latency (20 KIS text queries, `top_k=200`, measured at `ef=1024`, the
recall-passing operating point):

| lane | dim | p50 (ms) | p95 (ms) |
|---|---|---|---|
| siglip2 | 1152 | 16.97 | 18.19 |
| metaclip2 | 1024 | 14.85 | 16.64 |
| qwen3vl | 2048 | 21.25 | 51.08 |

Recall@200, HNSW vs exact-IP numpy baseline (`recall_set` / `recall_score`):

| lane | ef=256 | ef=512 | ef=1024 | NFR >= 0.95 |
|---|---|---|---|---|
| siglip2 | 0.794 / 0.798 | 0.928 / 0.932 | 0.969 / 0.971 | MET at ef=1024 |
| metaclip2 | 0.826 / 0.827 | 0.906 / 0.906 | 0.968 / 0.968 | MET at ef=1024 |
| qwen3vl | 0.952 / 0.959 | 0.978 / 0.985 | 0.985 / 0.992 | MET (from ef=256) |

**Verdict: the SS 6 recall@200 >= 0.95 NFR is DEMONSTRATED on stable 2.5.x** at
`efSearch=1024` for all three lanes (qwen3vl meets it from `ef=256`); the
deferral in 11.5 is resolved. All p95 latencies stay far under the SS 6
< 150 ms target even at `ef=1024`. (20-query sample, so per-lane recall varies
~0.005 run-to-run; both runs cleared 0.95 at ef=1024.)

**Operating-point caveat (RESOLVED 2026-06-05):** 2.5.x requires
`efSearch >= top_k`, so the proposal-01 `efSearch=128` default is **below**
`top_k=200` and is rejected by the server; it also under-shoots the recall NFR
(at ef=256 siglip2 is only 0.794). Resolved two ways: (1) the SS 3
`HnswParams.ef_search` default is **raised to 1024**, the validated
recall-passing point; (2) `search` clamps `ef = max(ef, top_k)` so any caller
default stays a valid operating point regardless of `top_k`. The eval that
produced the table above passed `ef_search` explicitly.

NFR summary (stable 2.5.27):

| NFR (SS 6) | Result (2.5.27) | Verdict |
|---|---|---|
| Ingest full proxy < 30 min | 64 s | PASS |
| Query p95 < 150 ms | 16.6-51.1 ms (121k, H200, ef=1024) | PASS (scale/hw caveat) |
| Recall@200 >= 0.95 | 0.968-0.985 at ef=1024 (all lanes) | PASS |
| Compatibility (pymilvus >= 2.5) | client 2.5.18, server 2.5.27 | PASS |

Box note: the 2.5.27 container is **left running** (it is the deployment); the
foreign GPU 0 workload was untouched; eval used GPU 1 and is freed.

## 12. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-06-05 | implementer (user-directed) | Created (Draft). Multi-vector keyframe collection (siglip2/metaclip2/qwen3vl floor), offline ingest from SPEC-0004 npy+manifest, online per-field ANN query + scalar filter, Submission adapter. Awaiting human approval before advancing past Draft. |
| 2026-06-05 | implementer (user-directed) | Open questions Q-a..Q-d RESOLVED (SS 9); status Draft -> Implementing (solo flow, CONTRIBUTING). Implemented `aic2026/index/{milvus_schema,models,milvus_store}.py` + `bin/index` CLI (ingest / ingest-all / search) + `MilvusBackend` single-lane Backend. Added `pymilvus[milvus_lite] >= 2.5` to the `index` extra and the dev group; regenerated `uv.lock` (pymilvus 3.0, milvus-lite 3.0). AC1-AC7 tested against real Milvus Lite (FLAT, CPU, offline), AC8 by inspection test; all green (17 new tests). SS 3 updated for `parse_video_id`, `MilvusBackend`, load-before-search, and the `--query-text`/`--encoder` search path; SS 7 updated for the `milvus_lite` sub-extra requirement. |
| 2026-06-05 | implementer (user-directed) | Added SS 11 lease-box integration evidence: first full-corpus standalone HNSW run (milvusdb/milvus v3.0-beta, pymilvus 3.0.0) on the H200 lease. Ingest 546 videos / 121457 rows in 58 s; per-lane query p50 11-12 ms / p95 12-13 ms; HNSW-vs-exact recall@200 0.84-0.93. Surfaced a frame_id/PK contract mismatch with the real SPEC-0004 manifests (per-video frame number, not a global id) and a v3.0-beta limitation (client efSearch ignored) blocking the >= 0.95 recall NFR; both with remediation. Eval artifacts banked to R2 `eval/milvus-proxy-20260605/`. This evidence is uncommitted pending review. |
| 2026-06-05 | implementer (user-directed) | Fixed the SPEC-0004 <-> SPEC-0006 frame_id/PK contract (SS 11.2 RESOLVED): the store now derives `video_id` from the source (npy filename stem or explicit `ingest(video_id=...)` arg) and composes a global primary key `pk = "<video_id>_<frame_id>"`; the per-video `frame_id` and `video_id` are kept as scalar fields. Removed `parse_video_id`. Added `pk` to `KeyframeMeta`/`Hit`; `hits_to_submissions` now carries `pk` as `Submission.frame_id`. Touched `aic2026/index/{milvus_schema,models,milvus_store,__init__}.py` + `bin/index` CLI (passes `--video` as `video_id`). Updated SS 1, SS 2.1, SS 3, SS 4, AC2, AC5 and the AC2/AC5 unit tests + the SPEC-0004-shaped fixture (per-video frame_id, video in the npy filename). All 18 Milvus Lite tests green (213 unit total). |
| 2026-06-05 | implementer (user-directed) | Added SS 11.8: stable Milvus 2.5.27 re-run (client pymilvus 2.5.18, lockfile pin 3.0.0 untouched/uncommitted) on the H200 lease. Re-ingested the **unmodified** SPEC-0004 `index/` tree via the fixed `bin/index ingest-all` (no `index_milvus/` patch) - 546 videos / 121457 rows in 64 s, validating the Part 1 fix on real data. `efSearch` is honoured on 2.5.x (server rejects `ef < top_k`; recall moves with ef, unlike the byte-identical beta sweep). Recall@200 >= 0.95 MET at `ef=1024` for all three lanes (siglip2 0.969, metaclip2 0.968, qwen3vl 0.985; qwen3vl from ef=256), p95 latency 16.6-51.1 ms - the 11.5 recall deferral is resolved. Flagged the `HnswParams.ef_search=128 < top_k=200` default as a follow-up (2.5.x requires ef >= top_k). Artifacts banked to R2 `eval/milvus-proxy-25x-20260605/`. Uncommitted pending review. |
| 2026-06-05 | implementer (user-directed) | Settled the SS 11.8 operating-point follow-up: raised the `HnswParams.ef_search` default 128 -> **1024** (the validated recall-passing point) and added an `ef = max(ef, top_k)` clamp in `MilvusKeyframeStore.search` so the standalone 2.5.x `ef >= top_k` constraint is always satisfied. Updated SS 3, SS 6 (latency + recall NFR at efSearch=1024), SS 10, and the SS 11.8 caveat (now RESOLVED). Code-only behaviour change in `milvus_schema.py` + `milvus_store.py`; Milvus Lite (FLAT) ignores ef so the unit tests are unaffected. |
