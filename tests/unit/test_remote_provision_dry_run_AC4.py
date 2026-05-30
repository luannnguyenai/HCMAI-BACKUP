# Proves SPEC-0024 AC4: `bin/remote provision --dry-run` exits 0, prints the
# planned push/restore/sync steps, and performs no ssh/push/R2 side effects.

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aic2026.cli.remote import app


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch: pytest.MonkeyPatch):
    """Any real ssh/push during a dry-run must fail loudly."""

    def _fail(*_a: object, **_k: object) -> object:
        raise AssertionError("provision --dry-run must not perform side effects")

    monkeypatch.setattr("aic2026.cli.remote.push_git_archive", _fail)
    monkeypatch.setattr("aic2026.cli.remote.ssh_exec", _fail)
    monkeypatch.setenv("R2_BUCKET", "test-bucket")
    yield


def test_provision_dry_run_prints_plan_AC4() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["provision", "--sha", "abcdef1234567890", "--dry-run"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "DRY-RUN provision plan" in out
    assert "git archive abcdef1" in out  # scoped push, short sha
    assert "uv sync --frozen --extra embedding" in out
    assert "env-cache/uv-$(uname -m)" in out  # arch-tagged uv cache restore
    assert "test-bucket" in out  # bucket wired from env


def test_provision_dry_run_weights_step_only_with_flag_AC4() -> None:
    runner = CliRunner()
    without = runner.invoke(app, ["provision", "--sha", "abcdef1234567890", "--dry-run"])
    assert "restore weights" not in without.output

    with_flag = runner.invoke(
        app, ["provision", "--sha", "abcdef1234567890", "--dry-run", "--restore-weights"]
    )
    assert "restore weights" in with_flag.output
    assert "s3://test-bucket/weights/" in with_flag.output


def test_provision_dry_run_no_env_skips_restore_AC4() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["provision", "--sha", "abcdef1234567890", "--dry-run", "--no-restore-env"]
    )
    assert result.exit_code == 0
    assert "restore uv cache" not in result.output
