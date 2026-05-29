# Proves SPEC-0022 AC8: cache-weights mirrors HF repos to weights/<repo>/ on
# R2 with a .cache-meta.json marker, is per-repo fault-tolerant, and raises
# only when nothing could be cached. snapshot_download is injected (no real
# huggingface_hub / network needed), R2 is a moto server.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("moto")
from moto.server import ThreadedMotoServer

from aic2026.remote.context import RunContext
from aic2026.remote.jobs.cache_weights import cache_weights

_BUCKET = "test-weights-bucket"


@pytest.fixture(scope="module")
def moto_server():
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def r2_env(monkeypatch: pytest.MonkeyPatch, moto_server: str):
    """Point the job's internal R2Client() at the moto server via env, and
    create a fresh bucket. Returns a verification client + bucket name."""
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
        job_name="cache-weights",
        git_sha="0" * 40,
        utc_ts="20260530T000000Z",
        local_root=tmp_path,
    )


def _fake_snapshot_factory(tmp_path: Path):
    """Return a snapshot_download stub that fabricates a small repo dir."""

    def _fake(repo_id: str, revision=None):
        d = tmp_path / "hf" / repo_id.replace("/", "__")
        (d / "nested").mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text('{"x": 1}', encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"\x00\x01\x02\x03")
        (d / "nested" / "tokenizer.json").write_text("{}", encoding="utf-8")
        return str(d)

    return _fake


def test_cache_weights_mirrors_to_weights_prefix_AC8(r2_env, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cache_weights(
        ctx,
        {"repos": "BAAI/bge-m3,vinai/PhoWhisper-large"},
        snapshot_download=_fake_snapshot_factory(tmp_path),
    )

    keys = r2_env.list("weights")
    # Each repo's files land under weights/<repo>/...
    assert "weights/BAAI/bge-m3/config.json" in keys
    assert "weights/BAAI/bge-m3/model.safetensors" in keys
    assert "weights/BAAI/bge-m3/nested/tokenizer.json" in keys
    assert "weights/BAAI/bge-m3/.cache-meta.json" in keys
    assert "weights/vinai/PhoWhisper-large/model.safetensors" in keys

    # The meta marker round-trips and records file count.
    meta = json.loads(r2_env.get_bytes("weights/BAAI/bge-m3/.cache-meta.json"))
    assert meta["repo"] == "BAAI/bge-m3"
    assert meta["n_files"] == 3

    # The per-run summary was written locally for the runner ledger.
    summary = (ctx.local_run_dir / "cached.jsonl").read_text(encoding="utf-8")
    assert summary.count("\n") == 2
    assert '"ok": true' in summary


def test_cache_weights_is_fault_tolerant_AC8(r2_env, tmp_path: Path) -> None:
    good = _fake_snapshot_factory(tmp_path)

    def _flaky(repo_id: str, revision=None):
        if repo_id == "bad/repo":
            raise OSError("simulated 404 from the hub")
        return good(repo_id)

    ctx = _ctx(tmp_path)
    # One bad id in the middle must not abort the batch.
    cache_weights(
        ctx,
        {"repos": "good/one,bad/repo,good/two"},
        snapshot_download=_flaky,
    )

    keys = r2_env.list("weights")
    assert any(k.startswith("weights/good/one/") for k in keys)
    assert any(k.startswith("weights/good/two/") for k in keys)
    assert not any(k.startswith("weights/bad/repo/") for k in keys)

    lines = [
        json.loads(line)
        for line in (ctx.local_run_dir / "cached.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_repo = {entry["repo"]: entry for entry in lines}
    assert by_repo["good/one"]["ok"] is True
    assert by_repo["bad/repo"]["ok"] is False
    assert "error" in by_repo["bad/repo"]


def test_cache_weights_all_fail_raises_AC8(r2_env, tmp_path: Path) -> None:
    def _always_fail(repo_id: str, revision=None):
        raise RuntimeError("nope")

    ctx = _ctx(tmp_path)
    with pytest.raises(RuntimeError, match="0/"):
        cache_weights(ctx, {"repos": "a/b,c/d"}, snapshot_download=_always_fail)
