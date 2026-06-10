# Proves SPEC-0028 AC1: check_prefixes over a moto-backed bucket where every
# required prefix holds >= 1 object returns ok=True with correct per-prefix
# counts and no missing prefixes.

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("moto")
from moto.server import ThreadedMotoServer

from aic2026.remote.preflight import check_prefixes

_BUCKET = "test-aic2026-preflight"


@pytest.fixture(scope="module")
def moto_server():
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def r2(monkeypatch: pytest.MonkeyPatch, moto_server: str):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=moto_server,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        region_name="us-east-1",
    )
    bucket = f"{_BUCKET}-{os.urandom(4).hex()}"
    s3.create_bucket(Bucket=bucket)
    from aic2026.remote.r2 import R2Client

    yield R2Client(
        endpoint_url=moto_server,
        access_key_id="fake",
        secret_access_key="fake",
        bucket=bucket,
    )


def _seed(r2, tmp_path: Path, prefix: str, n: int) -> None:
    src = tmp_path / prefix.replace("/", "_")
    src.mkdir(parents=True)
    for i in range(n):
        (src / f"obj_{i}.bin").write_bytes(b"x")
    r2.upload_dir(src, prefix)


def test_check_prefixes_all_present_ok_AC1(r2, tmp_path: Path) -> None:
    _seed(r2, tmp_path, "index/aic2025-proxy-3enc", 3)
    _seed(r2, tmp_path, "keyframes/aic2025-proxy", 5)

    result = check_prefixes(r2, ["index/aic2025-proxy-3enc", "keyframes/aic2025-proxy"])

    assert result.ok is True
    assert result.missing() == ()
    counts = {s.prefix: s.object_count for s in result.statuses}
    assert counts == {"index/aic2025-proxy-3enc": 3, "keyframes/aic2025-proxy": 5}
    assert all(s.present for s in result.statuses)
