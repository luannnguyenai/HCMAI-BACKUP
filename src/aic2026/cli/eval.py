# Implements SPEC-0001 SS 3.2 (CLI surface) and AC1, AC2, AC4.
"""`bin/eval` command-line interface.

Tier 1 implements the minimum surface needed for AC1: `--tasks`, `--system`,
`--output`, `--seed`, `--concurrency`. Options reserved for Tier 2/3
(`--baseline`, `--slice`, `--mode`, `--operator`, `--time-budget`,
`--dres-url`) are still listed in `--help` so users know the spec scope,
but invoking them exits with a clear "Tier 2/3" message.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from aic2026.harness.backend import StubBackend
from aic2026.harness.runner import EvalRunner, RunConfig, TaskLoadError, load_tasks
from aic2026.reporting.html import write_report_html
from aic2026.reporting.json_writer import write_metrics_json
from aic2026.reporting.provenance import write_readme

app = typer.Typer(
    add_completion=False,
    help=(
        "AIC2026 evaluation harness. Tier 1 ships AC1 + AC2 + AC4 of SPEC-0001 "
        "against a deterministic stub backend; AC3/AC5-AC8 land in follow-up tiers."
    ),
)

# Exit codes (kept consistent with SPEC-0001 AC3 and CI gating in AC7).
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_TASK_LOAD_FAILED = 2


def _logger() -> logging.Logger:
    return logging.getLogger("aic2026.cli.eval")


def _default_output_dir(system: str) -> Path:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return Path("eval-results") / system / ts


def _reject_tier_only_flag(flag: str, tier: str) -> None:
    typer.secho(
        f"ERROR: {flag} is reserved for {tier} of SPEC-0001 and is not wired in Tier 1. "
        "Open the corresponding follow-up PR; do not invoke this option for now.",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(EXIT_USAGE)


@app.command()
def eval_cmd(
    tasks: Annotated[
        Path,
        typer.Option(
            "--tasks",
            help="Path to a .jsonl mock-task set or a directory of *.jsonl.",
            exists=True,
            readable=True,
        ),
    ],
    system: Annotated[
        str,
        typer.Option(
            "--system",
            help="System version tag, e.g. 'v0.0.1-stub' or '<branch>+<short-sha>'.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Output directory. Default: eval-results/<system>/<timestamp>/.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed for the stub backend."),
    ] = 42,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            help="Parallel tasks. Tier 1 supports only 1; >1 lifts in Tier 2.",
            min=1,
            max=64,
        ),
    ] = 1,
    no_latency_sim: Annotated[
        bool,
        typer.Option(
            "--no-latency-sim",
            help="Skip the stub backend's simulated 10-30 ms latency. Useful in tests.",
        ),
    ] = False,
    baseline: Annotated[
        str | None,
        typer.Option("--baseline", help="[Tier 3 / AC6] reserved; not wired in Tier 1."),
    ] = None,
    slice_expr: Annotated[
        str | None,
        typer.Option("--slice", help="[Tier 3] reserved; not wired in Tier 1."),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="[Tier 2/3] reserved; not wired in Tier 1."),
    ] = None,
    operator: Annotated[
        str | None,
        typer.Option("--operator", help="[Tier 2/3] reserved; not wired in Tier 1."),
    ] = None,
    time_budget: Annotated[
        int | None,
        typer.Option("--time-budget", help="[Tier 2] reserved; not wired in Tier 1."),
    ] = None,
    dres_url: Annotated[
        str | None,
        typer.Option("--dres-url", help="[Tier 2 / AC3] reserved; not wired in Tier 1."),
    ] = None,
) -> None:
    """Run an evaluation against a deterministic stub backend.

    Per SPEC-0001 AC1, this produces `report.html` + `metrics.json` + `README.md`
    inside `--output` (or a timestamped subdirectory of `eval-results/`).
    """
    log = _logger()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Reject reserved-for-later flags loudly.
    if baseline is not None:
        _reject_tier_only_flag("--baseline", "Tier 3 / AC6")
    if slice_expr is not None:
        _reject_tier_only_flag("--slice", "Tier 3")
    if mode is not None:
        _reject_tier_only_flag("--mode", "Tier 2/3")
    if operator is not None:
        _reject_tier_only_flag("--operator", "Tier 2/3")
    if time_budget is not None:
        _reject_tier_only_flag("--time-budget", "Tier 2")
    if dres_url is not None:
        _reject_tier_only_flag("--dres-url", "Tier 2 / AC3")
    if concurrency != 1:
        _reject_tier_only_flag("--concurrency > 1", "Tier 2")

    out_dir = output or _default_output_dir(system)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading tasks from %s", tasks)
    try:
        mock_tasks = load_tasks(tasks)
    except TaskLoadError as exc:
        typer.secho(f"ERROR: failed to load tasks: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(EXIT_TASK_LOAD_FAILED) from None

    log.info("loaded %d tasks; running stub backend (seed=%d)", len(mock_tasks), seed)
    backend = StubBackend(seed=seed, simulate_latency=not no_latency_sim)
    config = RunConfig(
        system=system,
        tasks_path=tasks,
        output_dir=out_dir,
        seed=seed,
        concurrency=concurrency,
    )
    runner = EvalRunner(backend=backend, config=config)
    metrics = runner.run(mock_tasks)

    log.info("writing outputs to %s", out_dir)
    write_metrics_json(metrics, out_dir / "metrics.json")
    write_report_html(metrics, out_dir / "report.html")
    write_readme(metrics, out_dir, tasks_path=tasks)

    typer.echo(
        f"OK n={metrics.n_tasks} "
        f"R@1={metrics.overall.mean_r_at_1:.3f} "
        f"R@5={metrics.overall.mean_r_at_5:.3f} "
        f"R@10={metrics.overall.mean_r_at_10:.3f} "
        f"MRR={metrics.overall.mean_mrr:.3f} "
        f"p50={metrics.latency.p50_ms:.0f}ms p95={metrics.latency.p95_ms:.0f}ms"
    )
    typer.echo(f"   -> {out_dir.resolve()}")
    raise typer.Exit(EXIT_OK)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
