# Implements SPEC-0024 SS 3-4 (cache-env job) and AC2.
"""Mirror the uv wheel cache to R2 so a fresh lease's `uv sync` is a cache hit.

The expensive repeated cost on a new lease (ADR-0011: nothing under ~ survives)
is `uv sync` re-downloading GBs of CUDA/torch wheels. This job mirrors
`~/.cache/uv` to a STABLE, arch-tagged R2 prefix `env-cache/uv-<arch>/`. On the
next lease, `bin/remote provision` restores it before `uv sync`, turning the
sync into a near-instant cache hit.

Arch-tagged because the fleet is heterogeneous (x86_64 H100/H200, aarch64
Grace GB200): an x86 wheel cache and an aarch64 wheel cache must not clash.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

from aic2026.remote.context import RunContext
from aic2026.remote.registry import register

logger = logging.getLogger(__name__)

ENV_CACHE_PREFIX: str = "env-cache"


def _default_uv_cache_dir() -> Path:
    """Resolve the uv cache dir via `uv cache dir`, falling back to ~/.cache/uv."""
    try:
        out = subprocess.run(
            ["uv", "cache", "dir"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return Path.home() / ".cache" / "uv"


@register("cache-env")
def cache_env(
    ctx: RunContext,
    config: dict[str, Any],
    *,
    uv_cache_dir: Path | None = None,
) -> None:
    """Mirror the uv wheel cache to `env-cache/uv-<arch>/` on R2.

    Config:
        arch (str): override the arch tag (default: platform.machine()).

    `uv_cache_dir` is injectable for testing.
    """
    from aic2026.remote.r2 import R2Client

    arch = config.get("arch") or platform.machine()
    cache_dir = uv_cache_dir or _default_uv_cache_dir()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"uv cache dir not found: {cache_dir}")

    client = R2Client()
    prefix = f"{ENV_CACHE_PREFIX}/uv-{arch}"

    files = sorted(p for p in cache_dir.rglob("*") if p.is_file())
    n_bytes = 0
    for f in files:
        rel = f.relative_to(cache_dir).as_posix()
        client.upload_file(f, f"{prefix}/{rel}")
        n_bytes += f.stat().st_size

    meta = {
        "arch": arch,
        "n_files": len(files),
        "n_bytes": n_bytes,
        "run_id": ctx.run_id,
        "git_sha": ctx.git_sha,
        "source_dir": str(cache_dir),
    }
    client.put_bytes(
        f"{prefix}/.cache-meta.json",
        json.dumps(meta).encode("utf-8"),
        content_type="application/json",
    )

    ctx.local_run_dir.mkdir(parents=True, exist_ok=True)
    (ctx.local_run_dir / "cached_env.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "cached uv env -> s3://%s/%s (%d files, %.1f GB)",
        client.bucket,
        prefix,
        len(files),
        n_bytes / 1e9,
    )
