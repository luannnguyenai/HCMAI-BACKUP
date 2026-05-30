# Implements SPEC-0022 SS 3 (R2Client) and AC3.
"""Thin `boto3` wrapper for Cloudflare R2 (S3-compatible) per ADR-0011.

The class is intentionally small: it provides the four operations the runner
needs (upload_dir / list / download_dir / put_bytes / get_bytes) and nothing
else. Adding features that the spec does not require would invite drift from
the ADR's "no extra surface" rule.

Config is sourced from env vars (or explicit kwargs in tests):
    R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET,
    R2_REGION (defaults to "auto" - R2's preferred value).

The first call to `boto3.client(...)` is deferred to `__init__` and not done
at module import time so that `import aic2026.remote.r2` stays free of
network or env-validation surprises.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REGION: str = "auto"


def _required_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"required env var {name!r} is not set. See `.env.remote.example` "
            "and copy it to `.env.remote` (gitignored) before running."
        )
    return val


class R2Client:
    """Object storage on Cloudflare R2.

    All keys are bucket-relative (no leading slash). All `Path` arguments are
    local filesystem paths. The bucket is read from env at construction time
    and frozen on the instance.
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket: str | None = None,
        region: str | None = None,
    ) -> None:
        import boto3
        from botocore.config import Config

        self.bucket: str = bucket or _required_env("R2_BUCKET")
        self.endpoint_url: str = endpoint_url or _required_env("R2_ENDPOINT_URL")
        access = access_key_id or _required_env("R2_ACCESS_KEY_ID")
        secret = secret_access_key or _required_env("R2_SECRET_ACCESS_KEY")
        region = region or os.environ.get("R2_REGION", DEFAULT_REGION)

        # SPEC-0024 AC1: botocore >= 1.36 turns on default request/response
        # checksums that Cloudflare R2 does not fully support. The visible
        # symptom is `list_objects_v2` returning `NoSuchKey` against real R2
        # (uploads happen to survive because s3transfer self-corrects). Set
        # both knobs to "when_required" - Cloudflare's documented R2 fix.
        # https://developers.cloudflare.com/r2/examples/aws/boto3/
        r2_config = Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
        self._s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name=region,
            config=r2_config,
        )

    # --- single objects ----------------------------------------------------

    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        kwargs: dict[str, object] = {"Bucket": self.bucket, "Key": key, "Body": data}
        if content_type is not None:
            kwargs["ContentType"] = content_type
        self._s3.put_object(**kwargs)

    def get_bytes(self, key: str) -> bytes:
        resp = self._s3.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def upload_file(self, local_path: Path, key: str) -> None:
        """Stream a single file to `key` via boto3's managed transfer.

        Unlike `upload_dir` (which reads whole files into memory for a single
        PUT), this uses `s3.upload_file`, which streams from disk and does
        multipart automatically for large objects. Required for model weights
        whose safetensors shards exceed the ~5 GB single-PUT limit.
        """
        self._s3.upload_file(str(local_path), self.bucket, key)

    # --- directories -------------------------------------------------------

    def upload_dir(self, local: Path, prefix: str) -> list[str]:
        """Walk `local` recursively; upload every file under `<prefix>/<rel>`.

        Returns the bucket-relative keys written, sorted for determinism.
        """
        local = Path(local)
        if not local.exists():
            raise FileNotFoundError(f"upload_dir: local path missing: {local}")
        if not local.is_dir():
            raise NotADirectoryError(f"upload_dir: not a directory: {local}")
        prefix = prefix.rstrip("/")
        keys: list[str] = []
        for path in sorted(local.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local).as_posix()
            key = f"{prefix}/{rel}" if prefix else rel
            with path.open("rb") as fh:
                self._s3.put_object(Bucket=self.bucket, Key=key, Body=fh.read())
            keys.append(key)
        return keys

    def list(self, prefix: str) -> list[str]:
        """List every key under `prefix`. Paginates internally."""
        prefix = prefix.rstrip("/")
        paginator = self._s3.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                out.append(obj["Key"])
        return sorted(out)

    def download_dir(self, prefix: str, local: Path) -> list[Path]:
        """Mirror every object under `prefix` into `local`.

        Returns the sorted list of local paths written.
        """
        local = Path(local)
        local.mkdir(parents=True, exist_ok=True)
        prefix = prefix.rstrip("/")
        keys = self.list(prefix)
        written: list[Path] = []
        for key in keys:
            rel = key[len(prefix) :].lstrip("/")
            dst = local / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = self.get_bytes(key)
            dst.write_bytes(data)
            written.append(dst)
        return sorted(written)
