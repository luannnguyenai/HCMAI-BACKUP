# Implements SPEC-0022 SS 3-4 (remote-side job dispatcher).
"""`remote-job-exec` is the small entry-point the cluster runs to execute
one registered job. It is NEVER invoked from a developer laptop directly;
it is composed by `bin/remote run` and shipped over ssh.

Usage (built by `aic2026.cli.remote._build_remote_cmd`):

    remote-job-exec <job_name> <run_id> <remote_run_dir> [--config k=v ...]

The script reconstructs a RunContext (run_id is treated as authoritative;
git_sha is inferred from the run_id prefix and the local clone's HEAD),
imports all registered jobs, and invokes the resolved job.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated

import typer

from aic2026.remote.context import RunContext
from aic2026.remote.registry import resolve

app = typer.Typer(
    add_completion=False,
    help="SPEC-0022 remote-side job dispatcher. Invoked by `bin/remote run`.",
)


def _logger() -> logging.Logger:
    return logging.getLogger("aic2026.cli.remote_job_exec")


def _parse_config(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(f"--config expects key=value pairs; got {raw!r}")
        key, val = raw.split("=", 1)
        out[key.strip()] = val
    return out


def _git_sha_here() -> str:
    """Read the cluster-side clone's HEAD SHA. Fallback to 40-char zero."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "0" * 40


@app.command()
def exec_cmd(
    job_name: Annotated[str, typer.Argument(help="Registered job name.")],
    run_id: Annotated[str, typer.Argument(help="Authoritative run_id from `bin/remote run`.")],
    remote_run_dir: Annotated[
        Path,
        typer.Argument(help="Local-on-cluster directory to write outputs into."),
    ],
    config: Annotated[
        list[str] | None,
        typer.Option("--config", help="Free-form key=value pairs forwarded to the job."),
    ] = None,
) -> None:
    """Dispatch one registered job inside the cluster."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _logger()
    config = config or []

    # Side-effect: register all jobs.
    import aic2026.remote.jobs  # noqa: F401

    try:
        fn = resolve(job_name)
    except KeyError as exc:
        typer.secho(f"ERROR: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from None

    ctx = RunContext(
        run_id=run_id,
        git_sha=_git_sha_here(),
        job_name=job_name,
        started_at=datetime.now(UTC),
        local_run_dir=remote_run_dir,
        remote_run_dir=PurePosixPath(str(remote_run_dir)),
        r2_prefix=f"runs/{run_id}",
    )
    cfg = _parse_config(config)
    log.info("dispatching job %s (run_id=%s)", job_name, run_id)
    fn(ctx, cfg)
    log.info("job %s finished OK", job_name)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        # Typer's no-subcommand form is preferred for single-command apps.
        # Convert the decorated function above to act as the default callback.
        typer.run(exec_cmd)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
