# Implements SPEC-0022 SS 3-4 (SSH ops) and AC4.
"""Subprocess-based SSH wrappers.

We deliberately do **not** use paramiko / asyncssh. The user's existing
`~/.ssh/config` (Host alias `aic2026-gpu`, key auth, ControlMaster) is
strictly better-trodden than any Python SSH library, and the cost of
shelling out is invisible compared to network RTT to the cluster.

`ssh_exec` is the single primitive every other layer uses. On a non-zero
exit it raises `SSHError` carrying the remote stderr so the CLI can surface
something actionable.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

DEFAULT_SSH_HOST_ENV: str = "AIC2026_REMOTE_SSH_HOST"
DEFAULT_SSH_HOST: str = "aic2026-gpu"


class SSHError(RuntimeError):
    """ssh / scp exited non-zero. `returncode` and `stderr` are populated."""

    def __init__(self, message: str, *, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def resolve_host(explicit: str | None = None) -> str:
    """Pick the SSH host alias: arg > env > default."""
    if explicit:
        return explicit
    return os.environ.get(DEFAULT_SSH_HOST_ENV, DEFAULT_SSH_HOST)


def _format_env_prefix(env: dict[str, str]) -> str:
    """Turn `{"HF_TOKEN": "abc"}` into `HF_TOKEN=abc ...` for `ssh host "VAR=val cmd"`.

    Values are quoted with `shlex.quote` so secrets containing spaces or
    special chars survive the round-trip; they still appear in the remote
    process's argv (security-by-trust-of-the-cluster, NOT secret-from-the-OS).
    """
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))


def ssh_exec(
    host: str,
    cmd: str,
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `cmd` on `host` over ssh. Returns CompletedProcess; raises SSHError on failure.

    `env` is exported into the remote process's environment via `VAR=val cmd`
    (NOT written to remote disk). Use this for ephemeral credentials such as
    `HF_TOKEN` and the R2 keys - SPEC-0022 SS 6 (security).
    """
    env = env or {}
    remote_cmd = f"{_format_env_prefix(env)} {cmd}" if env else cmd
    proc = subprocess.run(
        ["ssh", host, remote_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise SSHError(
            f"ssh {host}: command failed (exit {proc.returncode}); stderr: {proc.stderr.strip()}",
            returncode=proc.returncode,
            stderr=proc.stderr,
        )
    return proc


def scp_push(
    host: str,
    local: Path,
    remote_path: str,
    *,
    recursive: bool = False,
    timeout: float | None = None,
) -> None:
    """Copy `local` to `host:remote_path`. Raises SSHError on failure."""
    args = ["scp"]
    if recursive:
        args.append("-r")
    args.extend([str(local), f"{host}:{remote_path}"])
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise SSHError(
            f"scp push to {host}:{remote_path} failed (exit {proc.returncode}); "
            f"stderr: {proc.stderr.strip()}",
            returncode=proc.returncode,
            stderr=proc.stderr,
        )


def scp_pull(
    host: str,
    remote_path: str,
    local: Path,
    *,
    recursive: bool = False,
    timeout: float | None = None,
) -> None:
    """Copy `host:remote_path` to `local`. Raises SSHError on failure."""
    args = ["scp"]
    if recursive:
        args.append("-r")
    args.extend([f"{host}:{remote_path}", str(local)])
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise SSHError(
            f"scp pull from {host}:{remote_path} failed (exit {proc.returncode}); "
            f"stderr: {proc.stderr.strip()}",
            returncode=proc.returncode,
            stderr=proc.stderr,
        )
