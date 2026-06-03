# Implements SPEC-0004 SS 3 (CLI surface).
"""`bin/embed` command-line interface.

Two subcommands:
  - `images`: walk a directory and write `.npy` + manifest (offline).
  - `text`:   one-shot encode of a string (debug surface; prints a few
              elements of the vector and its L2 norm).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from aic2026.embedding.base import Embedder
from aic2026.embedding.dummy import DummyEmbedder
from aic2026.embedding.extract import discover_images, extract_image_embeddings

app = typer.Typer(
    add_completion=False,
    help=(
        "AIC2026 embedding service. Offline image extraction (GH200, "
        "ADR-0003) and a small text-encode debug surface. Real backbones "
        "(SigLIP-2 et al.) require the `embedding` extra: "
        "`uv sync --extra embedding`."
    ),
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3

KNOWN_ENCODERS: tuple[str, ...] = ("dummy", "siglip2")


def _logger() -> logging.Logger:
    return logging.getLogger("aic2026.cli.embed")


def _resolve_encoder(name: str, dim: int) -> Embedder:
    name = name.lower()
    if name == "dummy":
        return DummyEmbedder(dim=dim)
    if name == "siglip2":
        try:
            from aic2026.embedding.siglip2 import SigLip2Embedder
        except ImportError as exc:  # pragma: no cover - exercised manually
            typer.secho(f"ERROR: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(EXIT_USAGE) from None
        try:
            return SigLip2Embedder()
        except ImportError as exc:
            typer.secho(f"ERROR: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(EXIT_USAGE) from None
    typer.secho(
        f"ERROR: unknown encoder {name!r}; known: {', '.join(KNOWN_ENCODERS)}.",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(EXIT_USAGE)


@app.command("images")
def images_cmd(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Directory containing *.jpg/jpeg/png/webp keyframes.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Output base path; writes <output>.npy + <output>.manifest.jsonl.",
        ),
    ],
    encoder: Annotated[
        str,
        typer.Option(
            "--encoder",
            help=f"Encoder to use; one of: {', '.join(KNOWN_ENCODERS)}.",
        ),
    ] = "dummy",
    dim: Annotated[
        int,
        typer.Option(
            "--dim",
            help="Output dimensionality; only honoured by `dummy`. SigLIP-2 is fixed at 1024.",
            min=1,
        ),
    ] = 64,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Per-call batch size.", min=1, max=1024),
    ] = 32,
) -> None:
    """Offline-extract image embeddings from a directory of keyframes."""
    log = _logger()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    emb = _resolve_encoder(encoder, dim)
    paths = discover_images(input_dir)
    log.info("found %d image(s) under %s", len(paths), input_dir)

    result = extract_image_embeddings(paths, emb, out=output, batch_size=batch_size)

    typer.echo(
        f"OK encoder={emb.model_id} n={result.n} dim={result.dim} "
        f"vectors={result.vectors_path} manifest={result.manifest_path}"
    )
    raise typer.Exit(EXIT_OK)


BENCH_ENCODERS: tuple[str, ...] = ("dummy", "siglip2", "metaclip2", "qwen3vl", "provided")


def _resolve_bench_encoder(
    name: str,
    *,
    device: str,
    dtype: str,
    provided_features: Path | None,
    qwen_out_dim: int | None,
    qwen_impl_src: str | None,
) -> Embedder:
    """Construct a bench encoder by name (heavy ones lazy-imported)."""
    name = name.lower()
    if name == "dummy":
        return DummyEmbedder(dim=64)
    if name == "siglip2":
        from aic2026.embedding.siglip2 import SigLip2Embedder

        return SigLip2Embedder(device=device, dtype=dtype)
    if name == "metaclip2":
        from aic2026.embedding.metaclip2 import MetaClip2Embedder

        return MetaClip2Embedder(device=device, dtype=dtype)
    if name == "qwen3vl":
        from aic2026.embedding.qwen3vl_embed import Qwen3VLEmbedder

        return Qwen3VLEmbedder(
            device=device, dtype=dtype, out_dim=qwen_out_dim, impl_src=qwen_impl_src
        )
    if name == "provided":
        from aic2026.embedding.provided_clip import ProvidedClipEmbedder

        if provided_features is None:
            raise typer.BadParameter("--provided-features is required for the 'provided' encoder")
        return ProvidedClipEmbedder.from_dir(provided_features, device=device, strict=False)
    raise typer.BadParameter(f"unknown bench encoder {name!r}; known: {', '.join(BENCH_ENCODERS)}")


def _load_queries(queries: Path, max_queries: int) -> list[str]:
    """Query texts: a dir of *kis*.txt (one query per file) or a newline file."""
    if queries.is_dir():
        files = sorted(queries.rglob("*kis*.txt")) or sorted(queries.rglob("*.txt"))
        out = [p.read_text(encoding="utf-8", errors="replace").strip() for p in files]
    else:
        out = [
            ln.strip()
            for ln in queries.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    out = [q for q in out if q]
    return out[:max_queries]


@app.command("bench")
def bench_cmd(
    kf_root: Annotated[Path, typer.Option("--kf-root", exists=True, file_okay=False)],
    queries: Annotated[Path, typer.Option("--queries", exists=True)],
    out: Annotated[Path, typer.Option("--out", help="Output dir for report + json.")],
    encoders: Annotated[
        str, typer.Option("--encoders", help=f"CSV of: {', '.join(BENCH_ENCODERS)}")
    ] = "siglip2,metaclip2,qwen3vl,provided",
    n_docs: Annotated[int, typer.Option("--n-docs", min=1)] = 20000,
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 5,
    max_queries: Annotated[int, typer.Option("--max-queries", min=1)] = 20,
    provided_features: Annotated[Path | None, typer.Option("--provided-features")] = None,
    qwen_out_dim: Annotated[int | None, typer.Option("--qwen-out-dim")] = None,
    qwen_impl_src: Annotated[
        str | None,
        typer.Option("--qwen-impl-src", help="Path to the cloned QwenLM/Qwen3-VL-Embedding repo."),
    ] = None,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    dtype: Annotated[str, typer.Option("--dtype")] = "float16",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Encoder bake-off (SPEC-0025): qualitative side-by-side + deployability."""
    import json

    from aic2026.eval.encoder_bench import (
        measure_deployability,
        run_qualitative,
        sample_keyframes,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in encoders.split(",") if n.strip()]
    query_texts = _load_queries(queries, max_queries)
    typer.echo(f"queries={len(query_texts)} encoders={names}")
    doc_paths = sample_keyframes(kf_root, n_docs, seed=seed)
    typer.echo(f"sampled {len(doc_paths)} keyframes from {kf_root}")

    built: dict[str, Embedder] = {}
    for name in names:
        try:
            built[name] = _resolve_bench_encoder(
                name,
                device=device,
                dtype=dtype,
                provided_features=provided_features,
                qwen_out_dim=qwen_out_dim,
                qwen_impl_src=qwen_impl_src,
            )
        except Exception as exc:
            typer.secho(f"WARN: skipping encoder {name!r}: {exc}", err=True, fg=typer.colors.YELLOW)
    if not built:
        typer.secho("ERROR: no encoders could be built", err=True, fg=typer.colors.RED)
        raise typer.Exit(EXIT_USAGE)

    deploy = [measure_deployability(enc, query_texts).as_dict() for enc in built.values()]
    (out / "deployability.json").write_text(json.dumps(deploy, indent=2), encoding="utf-8")
    for d in deploy:
        typer.echo(
            f"  {d['model_id']:>26}  dim={d['dim']:>4}  quant={d['quant']:>4}  "
            f"vram={d['vram_mb']}  p50={d['latency_p50_ms']}ms  fits={d['fits_5070_headroom']}"
        )

    report = out / "bench_report.html"
    run_qualitative(
        built, query_texts, doc_paths, top_k=top_k, out_html=report, batch_size=batch_size
    )
    typer.echo(f"OK wrote {report} + {out / 'deployability.json'}")
    raise typer.Exit(EXIT_OK)


@app.command("text")
def text_cmd(
    text: Annotated[
        str,
        typer.Option("--text", help="The string to encode."),
    ],
    encoder: Annotated[
        str,
        typer.Option(
            "--encoder",
            help=f"Encoder to use; one of: {', '.join(KNOWN_ENCODERS)}.",
        ),
    ] = "dummy",
    dim: Annotated[
        int,
        typer.Option(
            "--dim",
            help="Output dimensionality; only honoured by `dummy`.",
            min=1,
        ),
    ] = 64,
) -> None:
    """Debug-encode a single string; prints model_id, dim, norm and head."""
    emb = _resolve_encoder(encoder, dim)
    vec = emb.encode_text([text])
    import numpy as np

    norm = float(np.linalg.norm(vec[0]))
    head = ", ".join(f"{v:+.4f}" for v in vec[0][:8])
    typer.echo(f"OK encoder={emb.model_id} dim={emb.dim} norm={norm:.6f}\n   head[:8]=[{head}]")
    raise typer.Exit(EXIT_OK)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
