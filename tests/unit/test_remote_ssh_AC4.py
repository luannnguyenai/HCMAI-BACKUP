# Proves SPEC-0022 AC4: ssh_exec returns CompletedProcess on success;
# on non-zero exit it raises SSHError carrying stderr.

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aic2026.remote.ssh import (
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_HOST_ENV,
    SSHError,
    resolve_host,
    ssh_exec,
)


def _fake_completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_ssh_exec_success_returns_completed_process_AC4() -> None:
    fake = _fake_completed(0, stdout="hello\n", stderr="")
    with patch("aic2026.remote.ssh.subprocess.run", MagicMock(return_value=fake)) as run:
        result = ssh_exec("aic2026-gpu", "echo hello")
    assert result.returncode == 0
    assert result.stdout == "hello\n"
    args, _ = run.call_args
    # First positional is the argv; assert the ssh invocation shape.
    assert args[0][0] == "ssh"
    assert args[0][1] == "aic2026-gpu"
    assert args[0][2] == "echo hello"


def test_ssh_exec_failure_raises_with_stderr_AC4() -> None:
    fake = _fake_completed(2, stdout="", stderr="boom!\n")
    with (
        patch("aic2026.remote.ssh.subprocess.run", MagicMock(return_value=fake)),
        pytest.raises(SSHError) as exc_info,
    ):
        ssh_exec("aic2026-gpu", "nope")
    err = exc_info.value
    assert err.returncode == 2
    assert "boom!" in err.stderr
    assert "boom!" in str(err)


def test_ssh_exec_env_prefix_is_injected_AC4() -> None:
    fake = _fake_completed(0, stdout="", stderr="")
    with patch("aic2026.remote.ssh.subprocess.run", MagicMock(return_value=fake)) as run:
        ssh_exec("h", "do-stuff", env={"HF_TOKEN": "abc def", "R2_BUCKET": "b"})
    args, _ = run.call_args
    # Sorted env prefix, single quotes around the value with a space.
    remote_cmd = args[0][2]
    assert remote_cmd.startswith("HF_TOKEN='abc def' R2_BUCKET=b ")
    assert remote_cmd.endswith(" do-stuff")


def test_ssh_exec_check_false_returns_nonzero_AC4() -> None:
    """When `check=False`, ssh_exec must return (not raise) on non-zero exit."""
    fake = _fake_completed(7, stderr="warn")
    with patch("aic2026.remote.ssh.subprocess.run", MagicMock(return_value=fake)):
        result = ssh_exec("h", "x", check=False)
    assert result.returncode == 7


def test_resolve_host_precedence_AC4(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1) explicit beats env
    monkeypatch.setenv(DEFAULT_SSH_HOST_ENV, "from-env")
    assert resolve_host("explicit") == "explicit"
    # 2) env beats default
    assert resolve_host(None) == "from-env"
    # 3) default when nothing else
    monkeypatch.delenv(DEFAULT_SSH_HOST_ENV, raising=False)
    assert resolve_host(None) == DEFAULT_SSH_HOST
