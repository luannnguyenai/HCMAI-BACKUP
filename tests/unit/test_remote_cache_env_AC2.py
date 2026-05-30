# Proves SPEC-0024 AC2: cache-env mirrors the uv cache to env-cache/uv-<arch>/
# on R2 with a .cache-meta.json marker, arch-tagged. moto + injected cache dir.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("moto")
from moto.server import ThreadedMotoServer

from aic2026.remote.context import RunContext
from aic2026.remote.jobs.cache_env import cache_env

_BUCKET = "cache-env-bucket"


@pytest.fixture(scope="module")
def moto_server():
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def r2(monkeypatch: pytest.MonkeyPatch, moto_server: str):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("R2_ENDPOINT_URL", moto_server)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake")
    bucket = f"{_BUCKET}-{os.urandom(4).hex()}"
    monkeypatch.setenv("R2_BUCKET", bucket)
    import boto3

    boto3.client(
        "s3",
        endpoint_url=moto_server,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        region_name="us-east-1",
    ).create_bucket(Bucket=bucket)
    from aic2026.remote.r2 import R2Client

    return R2Client(
        endpoint_url=moto_server,
        access_key_id="fake",
        secret_access_key="fake",
        bucket=bucket,
    )


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext.build(
        job_name="cache-env", git_sha="0" * 40, utc_ts="20260530T000000Z", local_root=tmp_path
    )


def _fake_uv_cache(tmp_path: Path) -> Path:
    d = tmp_path / "uvcache"
    (d / "wheels").mkdir(parents=True)
    (d / "wheels" / "torch.whl").write_bytes(b"fake-wheel")
    (d / "index.json").write_text("{}", encoding="utf-8")
    return d


def test_cache_env_mirrors_arch_tagged_AC2(r2, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cache_env(ctx, {"arch": "x86_64"}, uv_cache_dir=_fake_uv_cache(tmp_path))

    keys = r2.list("env-cache")
    assert "env-cache/uv-x86_64/wheels/torch.whl" in keys
    assert "env-cache/uv-x86_64/index.json" in keys
    assert "env-cache/uv-x86_64/.cache-meta.json" in keys

    meta = json.loads(r2.get_bytes("env-cache/uv-x86_64/.cache-meta.json"))
    assert meta["arch"] == "x86_64"
    assert meta["n_files"] == 2


def test_cache_env_arch_isolation_AC2(r2, tmp_path: Path) -> None:
    """Two arch tags do not clash."""
    cache_env(_ctx(tmp_path), {"arch": "aarch64"}, uv_cache_dir=_fake_uv_cache(tmp_path))
    keys = r2.list("env-cache")
    assert any(k.startswith("env-cache/uv-aarch64/") for k in keys)
    assert not any(k.startswith("env-cache/uv-x86_64/") for k in keys)


def test_cache_env_missing_dir_raises_AC2(r2, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cache_env(_ctx(tmp_path), {"arch": "x86_64"}, uv_cache_dir=tmp_path / "nope")
