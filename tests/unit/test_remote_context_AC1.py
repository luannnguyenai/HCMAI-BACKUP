# Proves SPEC-0022 AC1: RunContext produces a stable run_id and round-trips.

from __future__ import annotations

import re

import pytest

from aic2026.remote.context import (
    DRYRUN_SENTINEL,
    RUN_ID_PATTERN,
    RunContext,
    make_run_id,
)

_SHA = "0" * 40


def test_run_id_format_matches_pattern_AC1() -> None:
    rid = make_run_id(_SHA, "extract-siglip", utc_ts="20260529T080942Z")
    assert rid == "0000000-extract-siglip-20260529T080942Z"
    assert RUN_ID_PATTERN.fullmatch(rid) is not None


def test_run_id_dryrun_sentinel_AC1() -> None:
    rid = make_run_id(_SHA, "extract-siglip", utc_ts=DRYRUN_SENTINEL)
    assert rid.endswith("-DRYRUN")
    assert RUN_ID_PATTERN.fullmatch(rid) is not None


def test_run_id_rejects_bad_sha_AC1() -> None:
    with pytest.raises(ValueError, match="git_sha"):
        make_run_id("not-hex", "extract-siglip", utc_ts="20260529T080942Z")


def test_run_id_rejects_bad_job_name_AC1() -> None:
    with pytest.raises(ValueError, match="job_name"):
        make_run_id(_SHA, "Extract Siglip!", utc_ts="20260529T080942Z")


def test_runcontext_build_and_roundtrip_AC1() -> None:
    ctx = RunContext.build(
        job_name="extract-siglip",
        git_sha=_SHA,
        utc_ts="20260529T080942Z",
    )
    assert ctx.run_id == "0000000-extract-siglip-20260529T080942Z"
    assert ctx.git_sha == _SHA
    assert ctx.r2_prefix == f"runs/{ctx.run_id}"
    assert not ctx.r2_prefix.startswith("/")

    restored = RunContext.model_validate_json(ctx.model_dump_json())
    assert restored.run_id == ctx.run_id
    assert restored.local_run_dir == ctx.local_run_dir
    assert restored.remote_run_dir == ctx.remote_run_dir
    assert restored.r2_prefix == ctx.r2_prefix


def test_runcontext_rejects_leading_slash_r2_prefix_AC1() -> None:
    ctx = RunContext.build(job_name="extract-siglip", git_sha=_SHA, utc_ts="20260529T080942Z")
    bad = ctx.model_dump()
    bad["r2_prefix"] = "/runs/foo"
    with pytest.raises(ValueError):
        RunContext.model_validate(bad)


def test_runcontext_rejects_malformed_run_id_AC1() -> None:
    ctx = RunContext.build(job_name="extract-siglip", git_sha=_SHA, utc_ts="20260529T080942Z")
    bad = ctx.model_dump()
    bad["run_id"] = "no-prefix-pattern"
    with pytest.raises(ValueError):
        RunContext.model_validate(bad)


def test_run_id_pattern_does_not_accept_uppercase_AC1() -> None:
    # The pattern is strictly lowercase hex + lowercase job name.
    assert RUN_ID_PATTERN.fullmatch("AAAAAAA-extract-siglip-DRYRUN") is None
    assert (
        re.fullmatch(
            r"^[0-9a-f]{7}-[a-z0-9_-]+-(\d{8}T\d{6}Z|DRYRUN)$", "0000000-extract-siglip-DRYRUN"
        )
        is not None
    )
