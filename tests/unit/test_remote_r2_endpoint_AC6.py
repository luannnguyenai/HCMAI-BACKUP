# Proves SPEC-0024 AC6: R2Client strips a bucket/path component from the
# endpoint URL. A pasted dashboard URL (".../<bucket>") otherwise makes
# list_objects_v2 return NoSuchKey against real Cloudflare R2 - the bug the
# first H200 lease hit, which moto could not reproduce (a mock ignores the
# endpoint path). This was first misattributed to botocore checksums.

from __future__ import annotations

import pytest

pytest.importorskip("boto3")

from aic2026.remote.r2 import R2Client, _normalize_endpoint


def test_normalize_endpoint_strips_bucket_path() -> None:
    assert (
        _normalize_endpoint("https://acct.r2.cloudflarestorage.com/aic2026-artifacts")
        == "https://acct.r2.cloudflarestorage.com"
    )


def test_normalize_endpoint_strips_nested_path() -> None:
    assert (
        _normalize_endpoint("https://acct.r2.cloudflarestorage.com/bucket/extra/")
        == "https://acct.r2.cloudflarestorage.com"
    )


def test_normalize_endpoint_leaves_bare_host() -> None:
    bare = "https://acct.r2.cloudflarestorage.com"
    assert _normalize_endpoint(bare) == bare


def test_r2client_strips_bucket_suffix_from_endpoint_AC6() -> None:
    client = R2Client(
        endpoint_url="https://acct.r2.cloudflarestorage.com/aic2026-artifacts",
        access_key_id="fake",
        secret_access_key="fake",
        bucket="aic2026-artifacts",
        region="auto",
    )
    assert client.endpoint_url == "https://acct.r2.cloudflarestorage.com"
