# Implements SPEC-0014 section 3 (CLI surface).
"""`bin/train` - C1 DiacriticBERT corpus build, head training, robustness eval.

Subcommands:
  - ``c1-corpus``: harvest public Vietnamese text -> noisy pairs Parquet.
  - ``c1-fit``:    train the projection head over frozen BGE-M3.
  - ``c1-eval``:   degradation@k sweep over a queries file.

The corpus + training steps need the ``train`` extra (`uv sync --extra train`);
heavy imports are deferred into each command so `--help` works without them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    add_completion=False,
    help=(
        "C1 DiacriticBERT (SPEC-0014): build the contrastive corpus, train the "
        "diacritic-robust head over frozen BGE-M3, and run the degradation@k "
        "robustness eval. Corpus + training need `uv sync --extra train`."
    ),
)

EXIT_OK = 0


def _logger() -> logging.Logger:
    return logging.getLogger("aic2026.cli.train")


def _configure_logging() -> None:
    """INFO for our logs; quiet the chatty HTTP/HF loggers (they flood corpus runs)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "datasets",
        "urllib3",
        "filelock",
        "fsspec",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command("c1-corpus")
def c1_corpus(
    out: Annotated[Path, typer.Option("--out", help="Output Parquet path.")],
    max_per_source: Annotated[
        int | None,
        typer.Option("--max-per-source", help="Cap rows harvested per HF dataset."),
    ] = None,
    k: Annotated[int, typer.Option("--k", help="Noisy variants per clean string.", min=1)] = 4,
    seed: Annotated[int, typer.Option("--seed", help="Determinism seed.")] = 0,
) -> None:
    """Build the contrastive corpus from the default public Vietnamese sources."""
    _configure_logging()
    from aic2026.train.diacritic_corpus import build_corpus

    res = build_corpus(out=out, k=k, max_per_source=max_per_source, seed=seed)
    typer.echo(
        f"OK clean={res.n_clean} pairs={res.n_pairs} out={res.out} "
        f"used={res.sources_used} skipped={res.sources_skipped}"
    )
    raise typer.Exit(EXIT_OK)


@app.command("c1-fit")
def c1_fit(
    pairs: Annotated[
        Path,
        typer.Option("--pairs", help="Corpus Parquet from c1-corpus.", exists=True),
    ],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Where to write head.pt + meta.")],
    backbone: Annotated[str, typer.Option("--backbone")] = "BAAI/bge-m3",
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 20_000,
    batch_size: Annotated[int, typer.Option("--batch-size", min=2)] = 64,
    lr: Annotated[float, typer.Option("--lr")] = 2e-4,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Train the DiacriticBERT projection head over a frozen backbone."""
    _configure_logging()
    from aic2026.train.diacritic_bert import TrainConfig, train_diacritic_head

    cfg = TrainConfig(
        backbone=backbone, max_steps=max_steps, batch_size=batch_size, lr=lr, seed=seed
    )
    res = train_diacritic_head(pairs, cfg, out_dir=out_dir)
    typer.echo(
        f"OK loss {res.initial_loss:.4f} -> {res.final_loss:.4f} "
        f"steps={res.steps} ckpt={res.checkpoint} meta={res.meta}"
    )
    raise typer.Exit(EXIT_OK)


@app.command("c1-eval")
def c1_eval(
    queries: Annotated[
        Path,
        typer.Option("--queries", help="Newline-delimited clean Vietnamese queries.", exists=True),
    ],
    k: Annotated[int, typer.Option("--k", min=1)] = 10,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Degradation@k sweep over the queries (DummyEmbedder baseline).

    The trained-head encoder (MaxSim retrieval) is a follow-up; this surface
    exercises the degradation harness and gives a single-vector baseline number.
    """
    from aic2026.embedding.dummy import DummyEmbedder
    from aic2026.eval.diacritic_robustness import degradation_at_k

    qs = [ln.strip() for ln in queries.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not qs:
        typer.secho("ERROR: no queries in file.", err=True, fg=typer.colors.RED)
        raise typer.Exit(2)
    res = degradation_at_k(qs, DummyEmbedder(dim=256), k=k, seed=seed)
    for key, val in res.items():
        typer.echo(f"{key:>12}: {val:.4f}")
    raise typer.Exit(EXIT_OK)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
