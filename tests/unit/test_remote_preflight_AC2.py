# Proves SPEC-0028 AC2: a missing/empty required prefix makes check_prefixes
# return ok=False with that prefix marked present=False, and require_prefixes
# raises PreflightError naming it and carrying the same PreflightResult.

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("moto")
from moto.server import ThreadedMotoServer

from aic2026.remote.preflight import PreflightError, check_prefixes, require_prefixes

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


def test_check_prefixes_flags_missing_AC2(r2, tmp_path: Path) -> None:
    _seed(r2, tmp_path, "index/present", 2)

    result = check_prefixes(r2, ["index/present", "keyframes/never-banked"])

    assert result.ok is False
    assert result.missing() == ("keyframes/never-banked",)
    by_prefix = {s.prefix: s for s in result.statuses}
    assert by_prefix["index/present"].present is True
    assert by_prefix["keyframes/never-banked"].present is False
    assert by_prefix["keyframes/never-banked"].object_count == 0


def test_require_prefixes_raises_naming_missing_AC2(r2, tmp_path: Path) -> None:
    _seed(r2, tmp_path, "index/present", 1)

    with pytest.raises(PreflightError) as excinfo:
        require_prefixes(r2, ["index/present", "keyframes/never-banked"])

    exc = excinfo.value
    assert "keyframes/never-banked" in str(exc)
    assert "bank-before-consume" in str(exc)
    assert exc.result.ok is False
    assert exc.result.missing() == ("keyframes/never-banked",)


def test_require_prefixes_returns_result_when_present_AC2(r2, tmp_path: Path) -> None:
    _seed(r2, tmp_path, "index/present", 4)
    result = require_prefixes(r2, ["index/present"])
    assert result.ok is True
    assert result.statuses[0].object_count == 4
