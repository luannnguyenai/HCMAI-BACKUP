# Implements SPEC-0022 SS 3-4 (ManifestEntry + ledger) and AC2.
"""Append-only run ledger backed by R2.

To avoid append races between concurrent jobs the ledger is **one object per
entry** under `manifest/<key>.json`, not a single JSONL file. `read_all()`
lists the prefix, fetches each object, and orders the result by `started_at`
ascending - the same order entries were appended in.

The schema is intentionally small: anything a job wants to remember
ad-hoc lives in `env` (a whitelisted, never-secrets dict captured at
launch). The eval-harness `metrics.json` schema (SPEC-0001 SS 3.3) is the
big-data home; this is the index for "what did we run, where do the blobs
live, did it succeed".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from aic2026.remote.r2 import R2Client

# All entries land under this single prefix; no per-job sub-prefixes (yet).
MANIFEST_PREFIX: str = "manifest"


class ManifestEntry(BaseModel):
    """One ledger row. Append via `append_to_r2`; read all via `read_all`."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    git_sha: str
    job_name: str
    started_at: datetime
    finished_at: datetime | None = None
    exit_code: int | None = None
    r2_prefix: str
    blobs: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Whitelisted env captured at launch (e.g. SLURM_JOB_ID). NEVER secrets.",
    )


def _entry_key(entry: ManifestEntry) -> str:
    """Sortable-by-time key for one entry. The `started_at` prefix gives us
    near-chronological list order without an additional index."""
    ts = entry.started_at.strftime("%Y%m%dT%H%M%S")
    return f"{MANIFEST_PREFIX}/{ts}-{entry.run_id}.json"


def append_to_r2(client: R2Client, entry: ManifestEntry) -> str:
    """Persist one entry. Returns the bucket-relative key written."""
    key = _entry_key(entry)
    body = entry.model_dump_json().encode("utf-8")
    client.put_bytes(key, body, content_type="application/json")
    return key


def read_all(client: R2Client, *, limit: int = 100) -> list[ManifestEntry]:
    """Read every entry under the manifest prefix, oldest-first.

    `limit` caps the number of entries returned (most-recent suffix); pass
    `limit=0` to disable the cap.
    """
    keys = client.list(MANIFEST_PREFIX)
    entries: list[ManifestEntry] = []
    for key in keys:
        data = client.get_bytes(key)
        entries.append(ManifestEntry.model_validate_json(data))
    entries.sort(key=lambda e: e.started_at)
    if limit > 0 and len(entries) > limit:
        return entries[-limit:]
    return entries
