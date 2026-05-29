# Implements SPEC-0022 SS 3 (RunContext) and AC1.
"""Provenance + paths for one remote-runner invocation.

`RunContext` is built once at the top of `bin/remote run`, threaded through
every layer (launcher, R2 upload, manifest append), and finally serialised
into the R2 ledger entry. Pinning `(git_sha, job_name, utc_ts)` makes any
artifact re-derivable from this triple.
"""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

# SPEC-0022 SS 5 AC1: stable run_id format.
RUN_ID_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{7}-[a-z0-9_-]+-(\d{8}T\d{6}Z|DRYRUN)$")

# Sentinel timestamp used by `--dry-run` so the printed plan is reproducible.
DRYRUN_SENTINEL: str = "DRYRUN"


def utc_now_compact() -> str:
    """`20260529T080942Z` - a filesystem-safe UTC timestamp."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def read_git_sha() -> str:
    """Full 40-char SHA at HEAD; raises if not in a git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    return result.stdout.strip()


def make_run_id(git_sha: str, job_name: str, *, utc_ts: str | None = None) -> str:
    """Compose the canonical run_id. `utc_ts=DRYRUN_SENTINEL` for dry-runs."""
    if not re.fullmatch(r"[0-9a-f]{7,40}", git_sha):
        raise ValueError(f"git_sha must be a hex SHA; got {git_sha!r}")
    if not re.fullmatch(r"[a-z0-9_-]+", job_name):
        raise ValueError(f"job_name must match [a-z0-9_-]+; got {job_name!r}. Rename the job.")
    ts = utc_ts or utc_now_compact()
    return f"{git_sha[:7]}-{job_name}-{ts}"


class RunContext(BaseModel):
    """All the provenance + paths needed to execute one job (SPEC-0022 SS 3)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    git_sha: str = Field(min_length=7, max_length=40)
    job_name: str
    started_at: datetime
    local_run_dir: Path
    remote_run_dir: PurePosixPath
    r2_prefix: str

    @field_validator("run_id")
    @classmethod
    def _run_id_pattern(cls, v: str) -> str:
        if not RUN_ID_PATTERN.fullmatch(v):
            raise ValueError(f"run_id {v!r} does not match {RUN_ID_PATTERN.pattern}")
        return v

    @field_validator("r2_prefix")
    @classmethod
    def _r2_prefix_no_leading_slash(cls, v: str) -> str:
        if v.startswith("/"):
            raise ValueError(f"r2_prefix must be bucket-relative; got {v!r}")
        return v

    @classmethod
    def build(
        cls,
        *,
        job_name: str,
        git_sha: str | None = None,
        utc_ts: str | None = None,
        local_root: Path | None = None,
        remote_root: PurePosixPath | None = None,
    ) -> RunContext:
        """Construct the canonical context for one `bin/remote run` invocation.

        Pass `utc_ts=DRYRUN_SENTINEL` for `--dry-run` so the printed plan is
        stable across runs.
        """
        sha = git_sha if git_sha is not None else read_git_sha()
        run_id = make_run_id(sha, job_name, utc_ts=utc_ts)
        local_root = local_root or Path("eval-results") / "remote"
        remote_root = remote_root or PurePosixPath("~/aic2026") / sha[:7] / "runs"
        return cls(
            run_id=run_id,
            git_sha=sha,
            job_name=job_name,
            started_at=datetime.now(UTC),
            local_run_dir=local_root / run_id,
            remote_run_dir=remote_root / run_id,
            r2_prefix=f"runs/{run_id}",
        )
