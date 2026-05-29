# Proves SPEC-0022 AC2: ManifestEntry append-and-read against moto R2.

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from aic2026.remote.manifest import ManifestEntry

# See test_remote_r2_AC3.py for why ThreadedMotoServer is used over mock_aws().
pytest.importorskip("moto")
from moto.server import ThreadedMotoServer

_BUCKET = "test-aic2026-artifacts"


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
    bucket = f"{_BUCKET}-{os.urandom(4).hex()}"
    s3.create_bucket(Bucket=bucket)
    from aic2026.remote.r2 import R2Client

    yield R2Client(
        endpoint_url=moto_server,
        access_key_id="fake",
        secret_access_key="fake",
        bucket=bucket,
    )


def _entry(rid: str, *, when: datetime, exit_code: int = 0) -> ManifestEntry:
    return ManifestEntry(
        run_id=rid,
        git_sha="abc1234",
        job_name="extract-siglip",
        started_at=when,
        finished_at=when + timedelta(seconds=42),
        exit_code=exit_code,
        r2_prefix=f"runs/{rid}",
        blobs=[f"runs/{rid}/v.npy"],
        env={},
    )


def test_append_then_read_round_trip_AC2(r2) -> None:
    from aic2026.remote.manifest import append_to_r2, read_all

    t0 = datetime(2026, 5, 29, 8, 0, 0, tzinfo=UTC)
    e1 = _entry("aaaaaaa-extract-siglip-20260529T080000Z", when=t0)
    append_to_r2(r2, e1)

    entries = read_all(r2)
    assert len(entries) == 1
    assert entries[0].run_id == e1.run_id
    assert entries[0].blobs == e1.blobs
    assert entries[0].exit_code == 0


def test_read_all_orders_by_started_at_ascending_AC2(r2) -> None:
    from aic2026.remote.manifest import append_to_r2, read_all

    t0 = datetime(2026, 5, 29, 8, 0, 0, tzinfo=UTC)
    e_old = _entry("aaaaaaa-extract-siglip-20260529T080000Z", when=t0)
    e_new = _entry(
        "bbbbbbb-extract-siglip-20260529T093000Z",
        when=t0 + timedelta(hours=1, minutes=30),
    )

    # Append in REVERSE chronological order; read_all must still sort.
    append_to_r2(r2, e_new)
    append_to_r2(r2, e_old)

    entries = read_all(r2)
    assert [e.run_id for e in entries] == [e_old.run_id, e_new.run_id]


def test_read_all_limit_keeps_most_recent_AC2(r2) -> None:
    from aic2026.remote.manifest import append_to_r2, read_all

    t0 = datetime(2026, 5, 29, 8, 0, 0, tzinfo=UTC)
    for i in range(5):
        append_to_r2(
            r2,
            _entry(f"{i:07x}-extract-siglip-20260529T08000{i}Z", when=t0 + timedelta(seconds=i)),
        )

    last2 = read_all(r2, limit=2)
    assert len(last2) == 2
    assert last2[0].run_id.endswith("3Z")
    assert last2[1].run_id.endswith("4Z")


def test_failed_run_still_appendable_AC2(r2) -> None:
    """A failed job still gets a ledger entry with exit_code != 0."""
    from aic2026.remote.manifest import append_to_r2, read_all

    t0 = datetime(2026, 5, 29, 8, 0, 0, tzinfo=UTC)
    bad = _entry("ccccccc-extract-siglip-20260529T080000Z", when=t0, exit_code=137)
    append_to_r2(r2, bad)

    [got] = read_all(r2)
    assert got.exit_code == 137
