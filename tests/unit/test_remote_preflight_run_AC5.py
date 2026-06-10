# Proves SPEC-0028 AC5: `bin/remote run --dry-run --require-prefix <p>` lists the
# required prefix in the plan and performs no R2 precondition call, and `run`
# with no --require-prefix performs no precondition call (existing flow intact).

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aic2026.cli.remote import app


@pytest.fixture(autouse=True)
def _no_real_launchers_or_r2(monkeypatch: pytest.MonkeyPatch):
    """A dry-run must not launch a job or touch R2 (incl. the preflight check).

    `RunContext.build()` legitimately shells out to `git rev-parse HEAD`, so we
    only trap launchers + R2, mirroring test_remote_cli_dry_run_AC6.
    """

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must not invoke launchers or R2")

    for target in (
        "aic2026.cli.remote.launch_srun",
        "aic2026.cli.remote.launch_sbatch",
        "aic2026.cli.remote.launch_ssh",
        "aic2026.cli.remote.launch_local",
        "aic2026.cli.remote.R2Client",
        "aic2026.cli.remote.append_to_r2",
    ):
        monkeypatch.setattr(target, _fail)
    yield


def test_dry_run_with_require_prefix_lists_and_no_r2_AC5() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "extract-siglip",
            "--launcher",
            "srun",
            "--dry-run",
            "--require-prefix",
            "keyframes/aic2025-proxy",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DRY-RUN plan" in result.output
    assert "keyframes/aic2025-proxy" in result.output
    assert "R2 preflight require_prefixes" in result.output


def test_dry_run_without_require_prefix_has_no_preflight_AC5() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "extract-siglip", "--launcher", "srun", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "require    = []" in result.output
    assert "R2 preflight require_prefixes" not in result.output
