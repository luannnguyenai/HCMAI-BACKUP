# Implements SPEC-0022 SS 3 (CLI surface) and AC6; SPEC-0028 SS 3-4 (preflight).
"""`bin/remote` command-line interface.

Subcommands per SPEC-0022 SS 4: setup, provision, run, pull, list, teardown.
The `run` subcommand supports `--dry-run` (AC6) which prints the planned actions
and exits 0 with no side effects (no ssh, no R2).

`preflight` (SPEC-0028) is the bank-before-consume guard (ADR-0016 rule c): it
verifies a job's required R2 prefixes exist and are non-empty before the job
runs. `run --require-prefix` wires the same check in as an opt-in guard.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from aic2026.remote.context import DRYRUN_SENTINEL, RunContext
from aic2026.remote.launchers import (
    LaunchResult,
    launch_local,
    launch_sbatch,
    launch_srun,
    launch_ssh,
)
from aic2026.remote.manifest import ManifestEntry, append_to_r2, read_all
from aic2026.remote.preflight import PreflightError, check_prefixes, require_prefixes
from aic2026.remote.r2 import R2Client
from aic2026.remote.registry import resolve
from aic2026.remote.ssh import SSHError, push_git_archive, resolve_host, ssh_exec

app = typer.Typer(
    add_completion=False,
    help=(
        "AIC2026 remote GPU job runner (SPEC-0022). Ship code to an "
        "ephemeral leased cluster, run a job, sync results to Cloudflare R2 "
        "(ADR-0011). Configure via `.env.remote` (see `.env.remote.example`)."
    ),
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_REMOTE = 4
EXIT_PRECONDITION = 5  # SPEC-0028: a required R2 prefix is missing/empty

KNOWN_LAUNCHERS: tuple[str, ...] = ("srun", "sbatch", "ssh", "local")
ALLOWED_REMOTE_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "R2_ENDPOINT_URL",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_REGION",
)

# Committed paths shipped to the box (SPEC-0024): code + lock + scripts, NOT
# the 208 MB of docs/papers PDFs. Keep in sync with what `uv sync` needs.
CODE_PATHS: tuple[str, ...] = (
    "src",
    "bin",
    "tests",
    "infra",
    "pyproject.toml",
    "uv.lock",
    "ruff.toml",
    ".python-version",
    "README.md",
)

# PATH prefix every remote command needs (uv installs to ~/.local/bin, which
# non-interactive sshd shells don't pick up). Surfaced as a constant so the
# dry-run plan and the executed command stay identical.
_REMOTE_PATH_PREFIX = 'export PATH="$HOME/.local/bin:$PATH"'


def _logger() -> logging.Logger:
    return logging.getLogger("aic2026.cli.remote")


def _ensure_jobs_loaded() -> None:
    """Trigger `@register` side effects."""
    import aic2026.remote.jobs  # noqa: F401


def _parse_config(items: list[str]) -> dict[str, str]:
    """Parse `--config key=value` flags into a dict (all values stay strings)."""
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(f"--config expects key=value pairs; got {raw!r}")
        key, val = raw.split("=", 1)
        out[key.strip()] = val
    return out


def _whitelisted_remote_env() -> dict[str, str]:
    """Pluck only the keys we deliberately forward to the cluster.

    Secrets are passed to the remote process at launch time via env; they are
    never written to remote disk (SPEC-0022 SS 6 Security).
    """
    return {k: os.environ[k] for k in ALLOWED_REMOTE_ENV_KEYS if k in os.environ}


# --- setup -----------------------------------------------------------------


@app.command()
def setup() -> None:
    """Validate `.env.remote` + ssh handshake + R2 reachability."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _logger()
    missing: list[str] = []
    for key in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        typer.secho(
            f"ERROR: missing env vars: {', '.join(missing)}. "
            f"Copy `.env.remote.example` -> `.env.remote` and fill it in.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(EXIT_CONFIG)

    host = resolve_host()
    log.info("ssh handshake to %s", host)
    try:
        ssh_exec(host, "true", timeout=15.0)
    except SSHError as exc:
        typer.secho(f"ERROR: ssh {host} failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(EXIT_REMOTE) from None

    log.info("R2 list bucket (%s)", os.environ["R2_BUCKET"])
    try:
        client = R2Client()
        _ = client.list("")  # cheap reachability check
    except Exception as exc:
        typer.secho(f"ERROR: R2 unreachable: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(EXIT_CONFIG) from None

    typer.echo(f"OK host={host} bucket={os.environ['R2_BUCKET']}")
    raise typer.Exit(EXIT_OK)


# --- provision -------------------------------------------------------------


def _uv_cache_restore_cmd(bucket: str) -> str:
    """Remote command to restore the uv wheel cache from R2 (SPEC-0024).

    Uses `uvx --from awscli aws s3 sync` (no awscli install needed, just uv).
    Maps R2_* -> AWS_* and sets the checksum-compat env vars (same R2 issue as
    R2Client). Arch is resolved on the box via `$(uname -m)`. Falls back to a
    no-op when the cache prefix is absent (first lease).
    """
    return (
        f"{_REMOTE_PATH_PREFIX} && "
        "AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY "
        "AWS_REQUEST_CHECKSUM_CALCULATION=when_required "
        "AWS_RESPONSE_CHECKSUM_VALIDATION=when_required "
        f'uvx --from awscli aws s3 sync "s3://{bucket}/env-cache/uv-$(uname -m)/" '
        '"$(uv cache dir)" --endpoint-url "$R2_ENDPOINT_URL" --only-show-errors '
        '|| echo "(no uv cache in R2 yet; uv sync will download wheels)"'
    )


def _uv_sync_cmd(dest: str) -> str:
    return f"{_REMOTE_PATH_PREFIX} && cd {dest} && uv sync --frozen --extra embedding"


@app.command()
def provision(
    sha: Annotated[
        str,
        typer.Option("--sha", help="Git SHA (or branch/tag) to provision on the cluster."),
    ],
    restore_env: Annotated[
        bool,
        typer.Option(
            "--restore-env/--no-restore-env",
            help="Restore the uv wheel cache from R2 before uv sync (fast path).",
        ),
    ] = True,
    restore_weights: Annotated[
        bool,
        typer.Option("--restore-weights", help="Also pull weights/<repo>/ from R2."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan and exit 0; no side effects."),
    ] = False,
) -> None:
    """Provision a fresh lease in one command: push code, restore uv cache from
    R2, `uv sync` (cache hit), optionally restore weights. Idempotent.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _logger()

    host = resolve_host()
    base = os.environ.get("AIC2026_REMOTE_BASE", "~/aic2026")
    short = sha[:7]
    dest = f"{base}/{short}"
    bucket = os.environ.get("R2_BUCKET", "<R2_BUCKET>")
    env = _whitelisted_remote_env()

    push_label = f"git archive {short} -- {' '.join(CODE_PATHS)} | ssh {host} 'tar -x -C {dest}'"
    restore_cmd = _uv_cache_restore_cmd(bucket)
    sync_cmd = _uv_sync_cmd(dest)
    weights_cmd = (
        f"{_REMOTE_PATH_PREFIX} && "
        "AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY "
        "AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required "
        f'uvx --from awscli aws s3 sync "s3://{bucket}/weights/" "{base}/weights/" '
        '--endpoint-url "$R2_ENDPOINT_URL" --only-show-errors'
    )

    if dry_run:
        typer.echo("DRY-RUN provision plan (no side effects):")
        typer.echo(f"  host   = {host}")
        typer.echo(f"  dest   = {dest}")
        typer.echo(f"  1. push: {push_label}")
        if restore_env:
            typer.echo(f"  2. restore uv cache: {restore_cmd}")
        typer.echo(f"  3. uv sync: {sync_cmd}")
        if restore_weights:
            typer.echo(f"  4. restore weights: {weights_cmd}")
        raise typer.Exit(EXIT_OK)

    try:
        log.info("pushing code to %s:%s (scoped git archive)", host, dest)
        push_git_archive(host, sha, list(CODE_PATHS), dest, timeout=120.0)
        if restore_env:
            log.info("restoring uv cache from R2 (if present)")
            ssh_exec(host, restore_cmd, env=env, timeout=1800.0)
        log.info("uv sync --frozen --extra embedding")
        ssh_exec(host, sync_cmd, timeout=1800.0)
        if restore_weights:
            log.info("restoring weights from R2")
            ssh_exec(host, weights_cmd, env=env, timeout=7200.0)
    except SSHError as exc:
        typer.secho(f"ERROR: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(EXIT_REMOTE) from None

    typer.echo(f"OK provisioned {host}:{dest}")
    raise typer.Exit(EXIT_OK)


# --- run -------------------------------------------------------------------


def _format_dry_run_plan(
    *,
    ctx: RunContext,
    host: str,
    launcher: str,
    remote_cmd: str,
    config: dict[str, str],
    require_prefix: list[str] | None = None,
) -> str:
    lines = [
        "DRY-RUN plan (no side effects):",
        f"  host       = {host}",
        f"  launcher   = {launcher}",
        f"  run_id     = {ctx.run_id}",
        f"  git_sha    = {ctx.git_sha}",
        f"  job_name   = {ctx.job_name}",
        f"  local_dir  = {ctx.local_run_dir}",
        f"  remote_dir = {ctx.remote_run_dir}",
        f"  r2_prefix  = {ctx.r2_prefix}",
        f"  config     = {config}",
        f"  require    = {require_prefix or []}",
        "  -- planned commands --",
    ]
    if require_prefix:
        lines.append(f"  R2 preflight require_prefixes (SPEC-0028): {require_prefix}")
    lines += [
        f"  ssh {host} 'mkdir -p {ctx.remote_run_dir}'",
        f"  {launcher}: {remote_cmd}",
        f"  R2 upload_dir {ctx.local_run_dir} -> s3://<bucket>/{ctx.r2_prefix}/",
        f"  R2 append_to_r2 manifest entry for {ctx.run_id}",
    ]
    return "\n".join(lines)


def _build_remote_cmd(ctx: RunContext, config: dict[str, str]) -> str:
    """The command the launcher runs on the remote.

    PATH is prepended so uv is found on the non-interactive ssh shell
    (uv installs to ~/.local/bin, which sshd shells don't pick up).
    Uses the SPEC-0024 `--run-id`/`--out` flags on remote-job-exec.
    """
    base = os.environ.get("AIC2026_REMOTE_BASE", "~/aic2026")
    repo_dir = f"{base}/{ctx.git_sha[:7]}"
    config_args = " ".join(f"--config {k}={v}" for k, v in config.items())
    return (
        'export PATH="$HOME/.local/bin:$PATH" && '
        f"cd {repo_dir} && "
        f"mkdir -p {ctx.remote_run_dir} && "
        f"uv run remote-job-exec {ctx.job_name} "
        f"--run-id {ctx.run_id} --out {ctx.remote_run_dir} {config_args}"
    )


@app.command("run")
def run_cmd(
    job: Annotated[str, typer.Argument(help="Registered job name; see `bin/remote list`.")],
    launcher: Annotated[
        str,
        typer.Option(
            "--launcher",
            help=f"One of: {', '.join(KNOWN_LAUNCHERS)}",
        ),
    ] = "srun",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the planned actions and exit 0 (no side effects)."),
    ] = False,
    config: Annotated[
        list[str] | None,
        typer.Option("--config", help="Free-form key=value passed to the job. Repeatable."),
    ] = None,
    require_prefix: Annotated[
        list[str] | None,
        typer.Option(
            "--require-prefix",
            help=(
                "Opt-in bank-before-consume guard (SPEC-0028 / ADR-0016): a "
                "bucket-relative R2 prefix that must exist and be non-empty "
                "before the job runs. Repeatable. Omitted -> no precondition check."
            ),
        ),
    ] = None,
) -> None:
    """Run a registered job on the cluster; upload outputs to R2; append the ledger."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _logger()
    _ensure_jobs_loaded()
    config = config or []
    require_prefix = require_prefix or []

    if launcher not in KNOWN_LAUNCHERS:
        typer.secho(
            f"ERROR: --launcher must be one of {', '.join(KNOWN_LAUNCHERS)}; got {launcher!r}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(EXIT_USAGE)

    try:
        resolve(job)
    except KeyError as exc:
        typer.secho(f"ERROR: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(EXIT_USAGE) from None

    cfg = _parse_config(config)
    host = resolve_host()

    # Dry-run uses the DRYRUN sentinel so the plan is reproducible across runs.
    ctx = RunContext.build(
        job_name=job,
        utc_ts=DRYRUN_SENTINEL if dry_run else None,
    )
    remote_cmd = _build_remote_cmd(ctx, cfg)

    if dry_run:
        typer.echo(
            _format_dry_run_plan(
                ctx=ctx,
                host=host,
                launcher=launcher,
                remote_cmd=remote_cmd,
                config=cfg,
                require_prefix=require_prefix,
            )
        )
        raise typer.Exit(EXIT_OK)

    # SPEC-0028 / ADR-0016 rule (c): opt-in bank-before-consume guard. Only runs
    # when the caller declared required prefixes; otherwise no R2 call is made
    # and `run` keeps its prior behaviour exactly.
    if require_prefix:
        log.info("preflight: require R2 prefixes %s", require_prefix)
        try:
            result = require_prefixes(R2Client(), require_prefix)
        except PreflightError as exc:
            typer.secho(f"ERROR: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(EXIT_PRECONDITION) from None
        for status in result.statuses:
            log.info("preflight OK %s (%d object(s))", status.prefix, status.object_count)

    log.info("dispatching job=%s run_id=%s via %s", job, ctx.run_id, launcher)
    ctx.local_run_dir.mkdir(parents=True, exist_ok=True)
    env = _whitelisted_remote_env()

    result: LaunchResult
    if launcher == "local":
        result = launch_local(remote_cmd, env=env)
    elif launcher == "ssh":
        result = launch_ssh(host, remote_cmd, env=env)
    elif launcher == "sbatch":
        result = launch_sbatch(host, remote_cmd, env=env)
    else:
        result = launch_srun(host, remote_cmd, env=env)

    log.info("launcher exit=%s", result.exit_code)
    if result.stdout:
        typer.echo(result.stdout)
    if result.stderr:
        typer.secho(result.stderr, err=True, fg=typer.colors.YELLOW)

    # Always upload + append, even on failure, so partial work is not lost.
    blobs: list[str] = []
    try:
        client = R2Client()
        if ctx.local_run_dir.exists() and any(ctx.local_run_dir.iterdir()):
            blobs = client.upload_dir(ctx.local_run_dir, ctx.r2_prefix)
        entry = ManifestEntry(
            run_id=ctx.run_id,
            git_sha=ctx.git_sha,
            job_name=ctx.job_name,
            started_at=ctx.started_at,
            finished_at=datetime.now(UTC),
            exit_code=result.exit_code,
            r2_prefix=ctx.r2_prefix,
            blobs=blobs,
            env={"SLURM_JOB_ID": result.slurm_job_id} if result.slurm_job_id else {},
        )
        append_to_r2(client, entry)
        typer.echo(f"OK run_id={ctx.run_id} blobs={len(blobs)} exit={result.exit_code}")
    except Exception as exc:
        typer.secho(f"WARNING: post-run R2 sync failed: {exc}", err=True, fg=typer.colors.RED)

    raise typer.Exit(result.exit_code)


# --- preflight -------------------------------------------------------------


@app.command("preflight")
def preflight_cmd(
    require: Annotated[
        list[str] | None,
        typer.Option(
            "--require",
            help=(
                "Bucket-relative R2 prefix that must exist and be non-empty. "
                "Repeatable. At least one is required."
            ),
        ),
    ] = None,
) -> None:
    """Verify required R2 prefixes exist and are non-empty (SPEC-0028).

    The executable form of ADR-0016 rule (c): a lease job may only consume
    inputs that already live in R2, never a box-local copy. Exits 0 when every
    prefix is present, 5 when any is missing/empty, 2 when no `--require` is given.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    require = require or []
    if not require:
        typer.secho(
            "ERROR: preflight needs at least one --require <prefix> to check.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(EXIT_USAGE)

    result = check_prefixes(R2Client(), require)
    for status in result.statuses:
        mark = "OK " if status.present else "MISSING"
        typer.echo(f"{mark} {status.prefix} ({status.object_count} object(s))")

    if not result.ok:
        typer.secho(
            "ERROR: R2 precondition failed (ADR-0016 bank-before-consume): "
            f"missing/empty prefix(es): {', '.join(result.missing())}. "
            "Bank the input(s) before running the job.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(EXIT_PRECONDITION)

    typer.echo(f"OK all {len(result.statuses)} required prefix(es) present")
    raise typer.Exit(EXIT_OK)


# --- pull ------------------------------------------------------------------


@app.command("pull")
def pull_cmd(
    run_id: Annotated[str, typer.Argument(help="run_id from the ledger; see `bin/remote list`.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Local destination dir; default `eval-results/remote/<run_id>/`.",
        ),
    ] = None,
) -> None:
    """Mirror an R2 prefix back into a local directory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dest = output or Path("eval-results") / "remote" / run_id
    client = R2Client()
    paths = client.download_dir(f"runs/{run_id}", dest)
    typer.echo(f"OK pulled {len(paths)} file(s) to {dest}")
    raise typer.Exit(EXIT_OK)


# --- list ------------------------------------------------------------------


@app.command("list")
def list_cmd(
    job: Annotated[
        str | None,
        typer.Option("--job", help="Filter to a job name."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=1000, help="Max entries to show."),
    ] = 20,
) -> None:
    """Show the most-recent N ledger entries (optionally filtered by job)."""
    client = R2Client()
    entries = read_all(client, limit=0)
    if job:
        entries = [e for e in entries if e.job_name == job]
    entries = entries[-limit:]
    for e in entries:
        typer.echo(
            f"{e.started_at.isoformat()} {e.run_id} job={e.job_name} "
            f"exit={e.exit_code} blobs={len(e.blobs)}"
        )
    raise typer.Exit(EXIT_OK)


# --- teardown --------------------------------------------------------------


@app.command()
def teardown(
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Required; refuses to run without it."),
    ] = False,
) -> None:
    """Delete `~/aic2026/` on the cluster (the lease box). R2 is untouched."""
    if not confirm:
        typer.secho(
            "ERROR: teardown deletes ~/aic2026 on the remote. Pass --confirm.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(EXIT_USAGE)
    host = resolve_host()
    base = os.environ.get("AIC2026_REMOTE_BASE", "~/aic2026")
    ssh_exec(host, f"rm -rf {base}", timeout=60.0)
    typer.echo(f"OK removed {host}:{base}")
    raise typer.Exit(EXIT_OK)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
