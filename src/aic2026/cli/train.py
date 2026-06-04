# Implements SPEC-0014 section 3 (CLI surface) + SS 6 (live demo).
"""`bin/train` - C1 DiacriticBERT corpus build, head training, robustness eval, demo.

Subcommands:
  - ``c1-corpus``: harvest public Vietnamese text -> noisy pairs Parquet.
  - ``c1-fit``:    train the projection head over frozen BGE-M3.
  - ``c1-eval``:   degradation@k sweep over a queries file.
  - ``c1-demo``:   live side-by-side demo (canned + interactive REPL).

The corpus + training + demo steps need the ``train`` extra
(`uv sync --extra train`); heavy imports are deferred into each command so
`--help` works without them.
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
    k: Annotated[
        int,
        typer.Option(
            "--k",
            help=(
                "Noisy variants per clean string. Default 0 means 'one of each NoiseMode' "
                "(v2 schedule: 7 modes covering diacritic + OCR noise)."
            ),
            min=0,
        ),
    ] = 0,
    seed: Annotated[int, typer.Option("--seed", help="Determinism seed.")] = 0,
) -> None:
    """Build the contrastive corpus from the default public Vietnamese sources."""
    _configure_logging()
    from aic2026.train.diacritic_corpus import build_corpus

    # k=0 (CLI default) -> let build_corpus pick "one of each NoiseMode"
    res = build_corpus(out=out, k=(k or None), max_per_source=max_per_source, seed=seed)
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


def _load_queries(queries_file: Path | None, build_heldout: int, exclude: Path | None) -> list[str]:
    """Resolve the eval query list: a file, or harvested held-out queries.

    Exactly one of ``--queries`` / ``--build-heldout`` must be provided.
    """
    if (queries_file is None) == (build_heldout <= 0):
        msg = "exactly one of --queries or --build-heldout (positive int) must be set"
        raise typer.BadParameter(msg)
    if queries_file is not None:
        qs = [
            ln.strip() for ln in queries_file.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if not qs:
            raise typer.BadParameter(f"no queries in {queries_file}")
        return qs
    from aic2026.eval.diacritic_robustness import build_heldout_queries

    return build_heldout_queries(build_heldout, exclude_corpus=exclude, seed=0)


@app.command("c1-eval")
def c1_eval(
    queries: Annotated[
        Path | None,
        typer.Option("--queries", help="Newline-delimited clean Vietnamese queries.", exists=True),
    ] = None,
    build_heldout: Annotated[
        int,
        typer.Option(
            "--build-heldout",
            help="Harvest N held-out queries from public sources (disjoint from --exclude).",
            min=0,
        ),
    ] = 0,
    exclude: Annotated[
        Path | None,
        typer.Option("--exclude", help="Training Parquet to exclude anchors from.", exists=True),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Path to head.pt; when set, runs the C1 ship-gate three-way comparison.",
            exists=True,
        ),
    ] = None,
    backbone: Annotated[str, typer.Option("--backbone")] = "BAAI/bge-m3",
    k: Annotated[int, typer.Option("--k", min=1)] = 10,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional path to write the result JSON."),
    ] = None,
) -> None:
    """Degradation@k sweep.

    Two modes:

    * **--checkpoint head.pt** (real ship-gate): loads frozen BGE-M3 + the
      trained head and runs the three-way comparison (C1 on vs raw BGE-M3
      MaxSim vs BGE-M3 dense). Needs the ``train`` extra + a GPU is helpful.
    * **No --checkpoint** (smoke): runs degradation@k against ``DummyEmbedder``
      for a CPU-friendly single-vector baseline that exercises the harness.
    """
    _configure_logging()
    qs = _load_queries(queries, build_heldout, exclude)

    if checkpoint is None:
        from aic2026.embedding.dummy import DummyEmbedder
        from aic2026.eval.diacritic_robustness import degradation_at_k

        res = degradation_at_k(qs, DummyEmbedder(dim=256), k=k, seed=seed)
        for key, val in res.items():
            typer.echo(f"{key:>12}: {val:.4f}")
        if out is not None:
            import json

            out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        raise typer.Exit(EXIT_OK)

    # Ship-gate path: load BGE-M3 + head and run the three-way comparison.
    from aic2026.eval.diacritic_robustness import compare_c1_vs_baselines
    from aic2026.eval.retrievers import load_head
    from aic2026.train.diacritic_bert import BgeM3Backbone

    typer.echo(f"loading backbone: {backbone}")
    bb = BgeM3Backbone(backbone)
    typer.echo(f"loading head: {checkpoint}")
    head = load_head(checkpoint)

    typer.echo(
        f"running degradation@{k} on {len(qs)} queries (c1_on / baseline_maxsim / baseline_dense)"
    )
    result = compare_c1_vs_baselines(qs, backbone=bb, head=head, k=k, seed=seed)
    sg = result["ship_gate"]

    def _line(label: str, block: dict[str, float]) -> None:
        per = "  ".join(f"{m:>11}={block[m]:.4f}" for m in result["modes"])
        typer.echo(f"{label:>16} overall={block['overall']:.4f}  {per}")

    _line("c1_on", result["c1_on"])
    _line("baseline_maxsim", result["baseline_maxsim"])
    _line("baseline_dense", result["baseline_dense"])
    typer.echo(
        f"ship_gate: target>={sg['target']:.2f}  "
        f"passes_absolute={sg['passes_absolute']}  "
        f"beats_maxsim={sg['beats_baseline_maxsim']}  "
        f"beats_dense={sg['beats_baseline_dense']}  "
        f"VERDICT={'PASS' if sg['passes_ship_gate'] else 'FAIL'}"
    )

    if out is not None:
        import json

        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        typer.echo(f"wrote {out}")
    raise typer.Exit(EXIT_OK)


@app.command("c1-demo")
def c1_demo(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help="Path to head.pt (the trained C1 head). Required.",
            exists=True,
        ),
    ],
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Training Parquet to exclude anchors from when sampling demo docs.",
            exists=True,
        ),
    ],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="One of: canned, interactive, both, tune.",
        ),
    ] = "canned",
    n_docs: Annotated[
        int,
        typer.Option(
            "--n-docs",
            help=(
                "Held-out Vietnamese docs to index (disjoint from --pairs). "
                "Default 2000 de-saturates the index so the mixed_ocr wins are "
                "visible; drop to ~300 for a faster (but more tie-heavy) run."
            ),
            min=10,
        ),
    ] = 2000,
    backbone: Annotated[str, typer.Option("--backbone")] = "BAAI/bge-m3",
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 3,
    useful_k: Annotated[
        int,
        typer.Option(
            "--useful-k",
            help=(
                "Rank threshold for a 'usable' result. A gold beyond top-useful-k "
                "is graceful-degradation, not a clean win. Drives the verdict."
            ),
            min=1,
        ),
    ] = 10,
    sweep_n: Annotated[
        int,
        typer.Option("--sweep-n", help="(tune mode) seeds 0..N-1 to sweep per example.", min=2),
    ] = 16,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Run a live C1 ship-gate demo.

    ``canned`` (default) prints the curated Vietnamese examples that exercise the
    failure modes C1 was trained to survive, with an honest verdict per example
    (C1 THẮNG RÕ / HÒA / C1 TRỤ TỐT HƠN / C1 THUA) judged against a top-``useful-k``
    usefulness lens, plus a tally.

    ``interactive`` drops into a REPL: the audience types a query + optional
    noise mode and sees the same side-by-side block.

    ``both`` runs canned first, then interactive.

    ``tune`` sweeps ``noise_seed`` per example and reports a recommended seed
    (the one giving the cleanest C1 win) - used to re-tune the canned set when
    the head changes. Needs the GPU; does not print the showcase.
    """
    _configure_logging()
    mode_norm = mode.strip().lower()
    if mode_norm not in {"canned", "interactive", "both", "tune"}:
        raise typer.BadParameter(
            f"--mode must be one of canned/interactive/both/tune; got {mode!r}"
        )

    from aic2026.eval.demo import (
        C1_LABEL_DEFAULT,
        CANNED_EXAMPLES,
        run_canned,
        run_interactive,
        tune_seeds,
    )
    from aic2026.eval.diacritic_robustness import build_heldout_queries
    from aic2026.eval.retrievers import (
        BgeM3DenseEmbedder,
        DenseRetriever,
        MaxSimRetriever,
        load_head,
    )
    from aic2026.train.diacritic_bert import BgeM3Backbone

    typer.echo(f"loading backbone: {backbone}")
    bb = BgeM3Backbone(backbone)
    typer.echo(f"loading head: {checkpoint}")
    head = load_head(checkpoint)

    typer.echo(f"sampling {n_docs} held-out docs (excluding training anchors)")
    doc_pool = build_heldout_queries(n_docs, exclude_corpus=pairs, seed=seed)

    c1_label = C1_LABEL_DEFAULT
    retrievers = {
        c1_label: MaxSimRetriever(bb, head=head),
        "Baseline MaxSim (BGE-M3 thô)": MaxSimRetriever(bb, head=None),
        "Baseline Dense (BGE-M3 vector trung bình)": DenseRetriever(BgeM3DenseEmbedder(bb)),
    }

    if mode_norm == "tune":
        typer.echo(f"== seed sweep (sweep_n={sweep_n}, useful_k={useful_k}) ==")
        tune_seeds(
            retrievers=retrievers,
            doc_pool=doc_pool,
            c1_label=c1_label,
            sweep_n=sweep_n,
            useful_k=useful_k,
        )
        raise typer.Exit(EXIT_OK)

    if mode_norm in ("canned", "both"):
        typer.echo(f"== canned showcase ({len(CANNED_EXAMPLES)} examples) ==")
        run_canned(
            retrievers=retrievers,
            doc_pool=doc_pool,
            c1_label=c1_label,
            top_k=top_k,
            useful_k=useful_k,
        )
    if mode_norm in ("interactive", "both"):
        run_interactive(retrievers=retrievers, doc_pool=doc_pool, top_k=max(top_k, 5))

    raise typer.Exit(EXIT_OK)


@app.command("c1-calibrate")
def c1_calibrate(
    queries: Annotated[
        Path | None,
        typer.Option(
            "--queries",
            help="Real clean queries (file or dir) - the AIC2025 query/ set.",
            exists=True,
        ),
    ] = None,
    pairs: Annotated[
        Path | None,
        typer.Option(
            "--pairs",
            help="Our training Parquet; anchors profiled + synthetically noised.",
            exists=True,
        ),
    ] = None,
    ocr: Annotated[
        Path | None,
        typer.Option(
            "--ocr",
            help="Real OCR/ASR output strings (file or dir) to match noise against.",
            exists=True,
        ),
    ] = None,
    max_strings: Annotated[int, typer.Option("--max", min=1)] = 5000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    out: Annotated[Path | None, typer.Option("--out", help="Optional JSON report path.")] = None,
) -> None:
    """Calibrate the C1 synthetic noise schedule against real corpus text (SPEC-0014 Q2).

    Profiles surface statistics of the real queries, our training anchors, our
    synthetic per-mode noise, and (optionally) real OCR output, then emits a
    distribution-match report + advisory flags pointing at which noise knob to
    revisit. At least one of --queries / --pairs / --ocr is required.
    """
    _configure_logging()
    if queries is None and pairs is None and ocr is None:
        raise typer.BadParameter("provide at least one of --queries / --pairs / --ocr")

    import json

    from aic2026.train.calibrate import (
        compare,
        load_strings,
        profile_synthetic_noise,
        profile_text,
    )

    real_query = None
    if queries is not None:
        qs = load_strings(queries)
        real_query = profile_text(qs[:max_strings]) if qs else None
        typer.echo(f"real queries: {len(qs)} strings")

    real_ocr = None
    if ocr is not None:
        os_ = load_strings(ocr)
        real_ocr = profile_text(os_[:max_strings]) if os_ else None
        typer.echo(f"real OCR: {len(os_)} strings")

    our_anchor = None
    synthetic = None
    if pairs is not None:
        from aic2026.train.diacritic_corpus import read_pairs

        rows = read_pairs(pairs)
        anchors = list({str(r["anchor_clean"]) for r in rows})[:max_strings]
        our_anchor = profile_text(anchors) if anchors else None
        synthetic = profile_synthetic_noise(anchors, seed=seed) if anchors else None
        typer.echo(f"our anchors: {len(anchors)} unique")

    report = compare(
        real_query=real_query, our_anchor=our_anchor, real_ocr=real_ocr, synthetic=synthetic
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"\nVERDICT: {report['verdict']}")
    for f in report["flags"]:  # type: ignore[union-attr]
        typer.echo(f"  - {f}")

    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"wrote {out}")
    raise typer.Exit(EXIT_OK)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
