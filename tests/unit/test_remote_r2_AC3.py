# Proves SPEC-0022 AC3: R2Client.upload_dir / list / download_dir against
# a moto-mocked S3 backend (R2 speaks the same API).

from __future__ import annotations

import os
from pathlib import Path

import pytest

# moto's ThreadedMotoServer gives us a real localhost HTTP endpoint that
# behaves like S3. We use it (rather than `mock_aws()` patching) because the
# R2Client always passes a custom `endpoint_url` - and moto v5's in-process
# patcher does not intercept custom endpoints.
pytest.importorskip("moto")
from moto.server import ThreadedMotoServer

_BUCKET = "test-aic2026-artifacts"


@pytest.fixture(scope="module")
def moto_server():
    """Module-scoped moto S3 server; one HTTP server for the whole test module."""
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def r2(monkeypatch: pytest.MonkeyPatch, moto_server: str):
    """A fresh bucket per test, wired to the module-scoped moto server.

    Strips HTTP(S)_PROXY env vars so boto3 does not route requests through
    Cursor's sandbox proxy (which would bypass moto entirely).
    """
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    for var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(var, raising=False)

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=moto_server,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        region_name="us-east-1",
    )
    # moto resets state between modules but not necessarily tests; use a
    # unique bucket name per test invocation to be safe.
    bucket = f"{_BUCKET}-{os.urandom(4).hex()}"
    s3.create_bucket(Bucket=bucket)
    from aic2026.remote.r2 import R2Client

    yield R2Client(
        endpoint_url=moto_server,
        access_key_id="fake",
        secret_access_key="fake",
        bucket=bucket,
    )


def _seed_dir(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("alpha", encoding="utf-8")
    (src / "b.bin").write_bytes(b"\x00\x01\x02")
    sub = src / "sub"
    sub.mkdir()
    (sub / "c.json").write_text('{"k": "v"}', encoding="utf-8")
    return src


def test_upload_dir_returns_sorted_keys_AC3(r2, tmp_path: Path) -> None:
    src = _seed_dir(tmp_path)
    keys = r2.upload_dir(src, "runs/abc")
    assert keys == [
        "runs/abc/a.txt",
        "runs/abc/b.bin",
        "runs/abc/sub/c.json",
    ]


def test_list_matches_uploaded_keys_AC3(r2, tmp_path: Path) -> None:
    src = _seed_dir(tmp_path)
    r2.upload_dir(src, "runs/xyz")
    assert r2.list("runs/xyz") == [
        "runs/xyz/a.txt",
        "runs/xyz/b.bin",
        "runs/xyz/sub/c.json",
    ]


def test_download_dir_byte_identical_AC3(r2, tmp_path: Path) -> None:
    src = _seed_dir(tmp_path)
    r2.upload_dir(src, "runs/rt")
    dst = tmp_path / "out"
    written = r2.download_dir("runs/rt", dst)
    assert (dst / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert (dst / "b.bin").read_bytes() == b"\x00\x01\x02"
    assert (dst / "sub" / "c.json").read_text(encoding="utf-8") == '{"k": "v"}'
    assert len(written) == 3


def test_upload_rejects_missing_dir_AC3(r2, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        r2.upload_dir(tmp_path / "does-not-exist", "runs/missing")


def test_upload_rejects_file_path_AC3(r2, tmp_path: Path) -> None:
    f = tmp_path / "not-a-dir.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        r2.upload_dir(f, "runs/bad")


def test_r2client_requires_env_when_kwargs_absent_AC3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without env or kwargs, construction fails with an actionable message."""
    for key in (
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    ):
        monkeypatch.delenv(key, raising=False)
    from aic2026.remote.r2 import R2Client

    with pytest.raises(RuntimeError, match=r"\.env\.remote"):
        R2Client()
    # silence unused-import warnings if any
    _ = os.environ
