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
