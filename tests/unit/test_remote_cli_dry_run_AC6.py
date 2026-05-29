# Proves SPEC-0022 AC6: `bin/remote run extract-siglip --dry-run` exits 0,
# prints the planned actions, and writes nothing via the launchers or R2.

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aic2026.cli.remote import app


@pytest.fixture(autouse=True)
def _no_real_launchers_or_r2(monkeypatch: pytest.MonkeyPatch):
    """If the dry-run path tries to launch a job or touch R2, fail loudly.

    We deliberately do NOT patch `subprocess.run` globally - `RunContext.build()`
    legitimately calls `git rev-parse HEAD` to embed the SHA in the printed
    plan. AC6's no-side-effects contract is about launchers + R2, not about
    reading the local git state.
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


def test_run_extract_siglip_dry_run_AC6() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "extract-siglip",
            "--launcher",
            "srun",
            "--dry-run",
            "--config",
            "input_dir=/scratch/sample_frames",
        ],
    )

    assert result.exit_code == 0, result.output
    # Plan content checks (AC6: prints planned ssh/srun/R2 actions).
    out = result.output
    assert "DRY-RUN plan" in out
    assert "launcher   = srun" in out
    assert "job_name   = extract-siglip" in out
    assert "-DRYRUN" in out  # sentinel run_id suffix in the plan
    assert "R2 upload_dir" in out
    assert "R2 append_to_r2 manifest entry" in out
    assert "input_dir" in out  # the --config pair is echoed back

    # AC6 sanity: nothing under eval-results/remote/ was created with the
    # DRYRUN sentinel suffix by this invocation.
    eval_remote = Path("eval-results/remote")
    if eval_remote.exists():
        for entry in eval_remote.iterdir():
            assert not entry.name.endswith("-DRYRUN"), f"dry-run unexpectedly created {entry}"


def test_run_unknown_launcher_exits_usage_AC6() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "extract-siglip", "--launcher", "warp-drive"])
    assert result.exit_code == 2
    assert "warp-drive" in result.output


def test_run_unknown_job_exits_usage_AC6() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "no-such-job", "--launcher", "srun", "--dry-run"])
    assert result.exit_code == 2
    assert "no-such-job" in result.output


def test_teardown_without_confirm_refuses_AC6() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["teardown"])
    assert result.exit_code == 2
    assert "--confirm" in result.output


def test_dry_run_local_launcher_also_side_effect_free_AC6() -> None:
    """The `--launcher local` path must also short-circuit on `--dry-run`.

    If the dry-run forgot to return early, `launch_local` would be invoked and
    the autouse fixture's `_fail` sentinel would convert the assertion into
    a non-zero exit code with the marker message in the captured output.
    """
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "extract-siglip",
            "--launcher",
            "local",
            "--dry-run",
            "--config",
            "input_dir=/tmp/x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DRY-RUN plan" in result.output
