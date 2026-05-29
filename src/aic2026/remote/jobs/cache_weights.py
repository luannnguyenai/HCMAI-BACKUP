# Implements SPEC-0022 SS 3 (jobs.cache-weights).
"""Mirror HuggingFace model snapshots to R2 under a stable `weights/` prefix.

Why: GPU leases are ephemeral (ADR-0011). Re-downloading the floor models
(SigLIP-2, Meta CLIP 2, InternVideo2, BGE-M3, PhoWhisper, Vintern, Qwen-VL-72B)
from HuggingFace on every fresh lease costs hours. This job downloads them
once and mirrors them to `s3://<bucket>/weights/<repo_id>/`, so future leases
restore from R2 in minutes.

Design notes:
- The model list is a CONFIG input (`repos=a,b,c`), not hardcoded deep in
  code, so a wrong/renamed repo id is fixed without a code change. The
  default below is the floor set; ids the implementer is less sure of are
  flagged inline.
- Each repo is wrapped in try/except so one bad id does not abort the batch.
- Uploads use `R2Client.upload_file` (streaming, multipart) because 72B
  safetensors shards exceed the in-memory single-PUT path.
- Output to a STABLE prefix `weights/<repo>/` (not `runs/<run_id>/`) so the
  cache location is predictable across leases. A per-repo `.cache-meta.json`
  marker makes `weights/` self-describing. A per-run summary is written into
  `ctx.local_run_dir` so the standard runner ledger still records the run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aic2026.remote.context import RunContext
from aic2026.remote.registry import register

logger = logging.getLogger(__name__)

# Floor models worth pre-staging. Override per-run with `--config repos=...`.
# NOTE: SigLIP-2 / Meta CLIP 2 are pulled by open_clip from timm-hosted repos;
# the timm id below is a best guess - verify against `dl.log` and override via
# config if it FAILs. The other five are confirmed ids.
DEFAULT_FLOOR_REPOS: tuple[str, ...] = (
    "BAAI/bge-m3",
    "vinai/PhoWhisper-large",
    "5CD-AI/Vintern-3B-beta",
    "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "timm/ViT-SO400M-16-SigLIP2-384",  # open_clip SigLIP-2 source (verify)
)

WEIGHTS_PREFIX: str = "weights"


def _iter_files(root: Path) -> list[Path]:
    """All real files under `root`, following symlinks (HF snapshots are
    symlink farms into the blob store)."""
    return sorted(p for p in root.rglob("*") if p.is_file())


@register("cache-weights")
def cache_weights(
    ctx: RunContext,
    config: dict[str, Any],
    *,
    snapshot_download: Any = None,
) -> None:
    """Download each HF repo and mirror it to `weights/<repo>/` on R2.

    Config:
        repos (str): comma-separated HF repo ids. Default = DEFAULT_FLOOR_REPOS.
        revision (str): optional single revision applied to all repos.

    Requires env: R2_* (for upload), HF_TOKEN (recommended; higher rate limit).

    `snapshot_download` is injectable for testing (defaults to the real
    `huggingface_hub.snapshot_download`, imported lazily so CI - which has no
    `embedding` extra - can still import this module).
    """
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as snapshot_download

    from aic2026.remote.r2 import R2Client

    repos_cfg = config.get("repos")
    repos = (
        [r.strip() for r in repos_cfg.split(",") if r.strip()]
        if repos_cfg
        else list(DEFAULT_FLOOR_REPOS)
    )
    revision = config.get("revision") or None

    client = R2Client()
    ctx.local_run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ctx.local_run_dir / "cached.jsonl"

    results: list[dict[str, Any]] = []
    with summary_path.open("w", encoding="utf-8") as summary:
        for repo in repos:
            entry: dict[str, Any] = {"repo": repo, "ok": False}
            try:
                logger.info("snapshot_download %s", repo)
                local_dir = Path(snapshot_download(repo, revision=revision))
                files = _iter_files(local_dir)
                n_bytes = sum(f.stat().st_size for f in files)
                prefix = f"{WEIGHTS_PREFIX}/{repo}"
                for f in files:
                    rel = f.relative_to(local_dir).as_posix()
                    client.upload_file(f, f"{prefix}/{rel}")
                meta = {
                    "repo": repo,
                    "revision": revision,
                    "n_files": len(files),
                    "n_bytes": n_bytes,
                    "run_id": ctx.run_id,
                    "git_sha": ctx.git_sha,
                }
                client.put_bytes(
                    f"{prefix}/.cache-meta.json",
                    json.dumps(meta).encode("utf-8"),
                    content_type="application/json",
                )
                entry.update(ok=True, n_files=len(files), n_bytes=n_bytes, prefix=prefix)
                logger.info(
                    "cached %s -> s3://%s/%s (%d files, %.1f GB)",
                    repo,
                    client.bucket,
                    prefix,
                    len(files),
                    n_bytes / 1e9,
                )
            except Exception as exc:  # one bad repo must not abort the batch
                entry["error"] = repr(exc)
                logger.warning("FAILED to cache %s: %r", repo, exc)
            results.append(entry)
            summary.write(json.dumps(entry) + "\n")

    n_ok = sum(1 for r in results if r["ok"])
    logger.info("cache-weights done: %d/%d repos cached", n_ok, len(repos))
    if n_ok == 0:
        raise RuntimeError(f"cache-weights cached 0/{len(repos)} repos; see {summary_path}")
