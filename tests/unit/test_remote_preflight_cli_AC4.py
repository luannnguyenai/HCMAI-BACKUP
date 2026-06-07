# Proves SPEC-0028 AC4: `bin/remote preflight` exits 0 when all --require
# prefixes are present, exits 5 (precondition) and names the absent prefix when
# one is missing, and exits 2 (usage) when no --require is given.

from __future__ import annotations

from typing import ClassVar

import pytest
from typer.testing import CliRunner

from aic2026.cli.remote import EXIT_PRECONDITION, EXIT_USAGE, app


class _FakeR2Client:
    """Patched in for `aic2026.cli.remote.R2Client`; no network.

    A class-level `present` set decides which prefixes resolve to >= 1 object.
    """

    present: ClassVar[set[str]] = set()

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def list(self, prefix: str) -> list[str]:
        return [f"{prefix}/obj_0.bin"] if prefix in type(self).present else []


@pytest.fixture
def patch_r2(monkeypatch: pytest.MonkeyPatch):
    def _set(present: set[str]) -> None:
        _FakeR2Client.present = present
        monkeypatch.setattr("aic2026.cli.remote.R2Client", _FakeR2Client)

    return _set


def test_preflight_all_present_exits_ok_AC4(patch_r2) -> None:
    patch_r2({"index/present", "keyframes/present"})
    runner = CliRunner()
    result = runner.invoke(
        app, ["preflight", "--require", "index/present", "--require", "keyframes/present"]
    )
    assert result.exit_code == 0, result.output
    assert "OK all 2 required prefix(es) present" in result.output


def test_preflight_missing_exits_precondition_and_names_it_AC4(patch_r2) -> None:
    patch_r2({"index/present"})
    runner = CliRunner()
    result = runner.invoke(
        app, ["preflight", "--require", "index/present", "--require", "keyframes/never-banked"]
    )
    assert result.exit_code == EXIT_PRECONDITION, result.output
    assert "keyframes/never-banked" in result.output
    assert "MISSING" in result.output


def test_preflight_no_require_exits_usage_AC4(patch_r2) -> None:
    patch_r2(set())
    runner = CliRunner()
    result = runner.invoke(app, ["preflight"])
    assert result.exit_code == EXIT_USAGE, result.output
    assert "--require" in result.output
