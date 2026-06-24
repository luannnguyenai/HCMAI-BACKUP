# Implements SPEC-0006 SS 3 (bin/index CLI surface).
"""`bin/index` command-line interface for the Milvus keyframe store.

Three subcommands:
  - `ingest`:     load one video's `<enc>/<video>.npy` + manifest pairs.
  - `ingest-all`: discover and load every video under `--index-root`.
  - `search`:     ANN one dense field from a `--query-npy` or, via an injected
                  encoder, a `--query-text`; print ranked hits.

`--index-root` points at an R2-mirrored local tree (`<enc>/<video>.npy` +
`<video>.manifest.jsonl`); per-field paths are derived by the SPEC-0004 /
ADR-0011 convention. The CLI is thin: it parses paths and prints; all schema /
ingest / query logic lives in `aic2026.index.milvus_store`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from aic2026.index.milvus_schema import FLOOR_FIELDS
from aic2026.index.milvus_store import (
    DEFAULT_COLLECTION,
    DEFAULT_TOP_K,
    EncoderSource,
    MilvusKeyframeStore,
)

app = typer.Typer(
    add_completion=False,
    help=(
        "AIC2026 Milvus keyframe store (SPEC-0006). Offline ingest of the "
        "SPEC-0004 .npy + manifest pairs into one multi-vector collection, and "
        "online per-field ANN query. Dev/CI use Milvus Lite (a local .db path); "
        "the real HNSW build uses standalone (http://host:19530). Requires the "
        "`index` extra: `uv sync --extra index`."
    ),
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3

_DEFAULT_FIELDS_CSV = ",".join(f.name for f in FLOOR_FIELDS)


def _logger() -> logging.Logger:
    return logging.getLogger("aic2026.cli.index")


def _parse_fields_csv(fields_csv: str) -> tuple[str, ...]:
    names = tuple(n.strip() for n in fields_csv.split(",") if n.strip())
    if not names:
        raise typer.BadParameter("--fields must list at least one encoder name")
    return names


def _selected_fields(fields_csv: str) -> tuple:
    """Subset FLOOR_FIELDS to the requested names, preserving declared order."""
    wanted = set(_parse_fields_csv(fields_csv))
    by_name = {f.name: f for f in FLOOR_FIELDS}
    unknown = wanted - set(by_name)
    if unknown:
        raise typer.BadParameter(f"unknown field(s) {sorted(unknown)}; known: {', '.join(by_name)}")
    return tuple(f for f in FLOOR_FIELDS if f.name in wanted)


def _video_sources(index_root: Path, video: str, field_names: tuple[str, ...]) -> dict:
    """Per-field `EncoderSource` for one video under `<index_root>/<enc>/`."""
    sources: dict[str, EncoderSource] = {}
    for name in field_names:
        npy = index_root / name / f"{video}.npy"
        manifest = index_root / name / f"{video}.manifest.jsonl"
        if not npy.exists():
            raise typer.BadParameter(f"missing vectors for field {name!r}: {npy}")
        if not manifest.exists():
            raise typer.BadParameter(f"missing manifest for field {name!r}: {manifest}")
        sources[name] = EncoderSource(vectors=npy, manifest=manifest)
    return sources


def _discover_videos(index_root: Path, field_names: tuple[str, ...]) -> list[str]:
    """Video stems present (as `<video>.npy`) under the first field's dir."""
    first = field_names[0]
    field_dir = index_root / first
    if not field_dir.is_dir():
        raise typer.BadParameter(f"index-root has no {first!r} dir: {field_dir}")
    return sorted(p.stem for p in field_dir.glob("*.npy"))


@app.command("ingest")
def ingest_cmd(
    uri: Annotated[str, typer.Option("--uri", help="Milvus Lite .db path or http endpoint.")],
    index_root: Annotated[
        Path,
        typer.Option("--index-root", exists=True, file_okay=False, help="R2-mirrored tree root."),
    ],
    video: Annotated[str, typer.Option("--video", help="Video stem, e.g. L25_V011.")],
    collection: Annotated[str, typer.Option("--collection")] = DEFAULT_COLLECTION,
    fields: Annotated[
        str, typer.Option("--fields", help="CSV of encoder fields.")
    ] = _DEFAULT_FIELDS_CSV,
    metadata: Annotated[
        Path | None,
        typer.Option("--metadata", help="Optional scalar jsonl keyed by frame_id."),
    ] = None,
) -> None:
    """Ingest one video's per-field npy + manifest pairs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selected = _selected_fields(fields)
    store = MilvusKeyframeStore(uri=uri, collection=collection, fields=selected)
    sources = _video_sources(index_root, video, tuple(f.name for f in selected))
    result = store.ingest(sources, video_id=video, metadata=metadata)
    typer.echo(
        f"OK collection={result.collection} video={video} "
        f"n_rows={result.n_rows} fields={','.join(result.fields_loaded)}"
    )
    raise typer.Exit(EXIT_OK)


@app.command("ingest-all")
def ingest_all_cmd(
    uri: Annotated[str, typer.Option("--uri")],
    index_root: Annotated[Path, typer.Option("--index-root", exists=True, file_okay=False)],
    collection: Annotated[str, typer.Option("--collection")] = DEFAULT_COLLECTION,
    fields: Annotated[str, typer.Option("--fields")] = _DEFAULT_FIELDS_CSV,
    metadata: Annotated[Path | None, typer.Option("--metadata")] = None,
) -> None:
    """Ingest every video discovered under `--index-root`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _logger()
    selected = _selected_fields(fields)
    field_names = tuple(f.name for f in selected)
    store = MilvusKeyframeStore(uri=uri, collection=collection, fields=selected)
    videos = _discover_videos(index_root, field_names)
    log.info("discovered %d video(s) under %s", len(videos), index_root)
    total = 0
    for video in videos:
        sources = _video_sources(index_root, video, field_names)
        result = store.ingest(sources, video_id=video, metadata=metadata)
        total += result.n_rows
        log.info("ingested %s (%d rows)", video, result.n_rows)
    typer.echo(
        f"OK collection={collection} videos={len(videos)} n_rows={total} "
        f"fields={','.join(field_names)}"
    )
    raise typer.Exit(EXIT_OK)


@app.command("search")
def search_cmd(
    uri: Annotated[str, typer.Option("--uri")],
    field: Annotated[str, typer.Option("--field", help="Dense field to query.")],
    collection: Annotated[str, typer.Option("--collection")] = DEFAULT_COLLECTION,
    query_npy: Annotated[
        Path | None,
        typer.Option("--query-npy", help="(nq, dim) or (dim,) float32 .npy query."),
    ] = None,
    query_text: Annotated[
        str | None,
        typer.Option("--query-text", help="Text encoded via --encoder into a query vector."),
    ] = None,
    encoder: Annotated[
        str, typer.Option("--encoder", help="Encoder for --query-text (only `dummy` is CI-safe).")
    ] = "dummy",
    dim: Annotated[int, typer.Option("--dim", min=1, help="dummy encoder output dim.")] = 64,
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = DEFAULT_TOP_K,
    expr: Annotated[
        str | None, typer.Option("--expr", help="Scalar filter, e.g. \"video_id == 'L25_V011'\".")
    ] = None,
) -> None:
    """Search one dense field; print ranked hits (frame_id, score)."""
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if (query_npy is None) == (query_text is None):
        typer.secho(
            "ERROR: provide exactly one of --query-npy or --query-text.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(EXIT_USAGE)

    if query_npy is not None:
        if not query_npy.exists():
            typer.secho(f"ERROR: query npy not found: {query_npy}", err=True, fg=typer.colors.RED)
            raise typer.Exit(EXIT_NOT_FOUND)
        queries = np.load(query_npy).astype(np.float32, copy=False)
    else:
        queries = _encode_query_text(query_text or "", encoder=encoder, dim=dim)

    store = MilvusKeyframeStore(uri=uri, collection=collection)
    results = store.search(field, queries, top_k=top_k, expr=expr)
    for qi, hits in enumerate(results):
        typer.echo(f"query[{qi}] field={field} hits={len(hits)}")
        for hit in hits:
            typer.echo(f"  {hit.rank:>4}  {hit.score:+.4f}  {hit.pk}  ({hit.video_id})")
    raise typer.Exit(EXIT_OK)


def _encode_query_text(text: str, *, encoder: str, dim: int):
    """Encode one query string into a (1, dim) vector via an injected encoder."""
    import numpy as np

    name = encoder.lower()
    if name == "dummy":
        from aic2026.embedding.dummy import DummyEmbedder

        emb = DummyEmbedder(dim=dim)
    else:
        # Real encoders need the `embedding` extra; resolve lazily and surface
        # the extra hint on failure (mirrors bin/embed).
        try:
            from aic2026.cli.embed import _resolve_encoder
        except ImportError as exc:  # pragma: no cover - exercised manually
            typer.secho(f"ERROR: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(EXIT_USAGE) from None
        emb = _resolve_encoder(name, dim)
    return np.asarray(emb.encode_text([text]), dtype=np.float32)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
