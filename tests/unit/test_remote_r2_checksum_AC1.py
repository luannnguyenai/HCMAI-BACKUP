# Proves SPEC-0024 AC1: R2Client is built with the checksum-compat Config that
# fixes ListObjectsV2 NoSuchKey against real Cloudflare R2, and list() still
# works against moto (regression guard - moto does not reproduce the R2 bug).

from __future__ import annotations

import os

import pytest

pytest.importorskip("moto")
from moto.server import ThreadedMotoServer

from aic2026.remote.r2 import R2Client


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
    import boto3

    bucket = f"chk-{os.urandom(4).hex()}"
    boto3.client(
        "s3",
        endpoint_url=moto_server,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        region_name="us-east-1",
    ).create_bucket(Bucket=bucket)
    return R2Client(
        endpoint_url=moto_server,
        access_key_id="fake",
        secret_access_key="fake",
        bucket=bucket,
    )


def test_r2client_uses_when_required_checksums_AC1(r2) -> None:
    cfg = r2._s3.meta.config
    assert cfg.request_checksum_calculation == "when_required"
    assert cfg.response_checksum_validation == "when_required"


def test_list_works_against_moto_AC1(r2) -> None:
    r2.put_bytes("weights/foo/a.txt", b"hi")
    r2.put_bytes("weights/foo/b.txt", b"yo")
    assert r2.list("weights") == ["weights/foo/a.txt", "weights/foo/b.txt"]


def test_provision_restore_uses_lowercase_checksum_values_AC1() -> None:
    # botocore matches these values case-sensitively; uppercase WHEN_REQUIRED is
    # silently ignored, leaving the R2-incompatible default on. The provision
    # `aws s3 sync` restore commands must emit lowercase `when_required`.
    from aic2026.cli.remote import _uv_cache_restore_cmd

    cmd = _uv_cache_restore_cmd("some-bucket")
    assert "AWS_REQUEST_CHECKSUM_CALCULATION=when_required" in cmd
    assert "AWS_RESPONSE_CHECKSUM_VALIDATION=when_required" in cmd
    assert "WHEN_REQUIRED" not in cmd
