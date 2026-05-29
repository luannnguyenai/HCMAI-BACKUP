# Implements SPEC-0022 SS 3-4 (launchers).
"""Four launchers for `bin/remote run --launcher ...`.

A launcher is a free function that turns a (host, run_id, command, env) tuple
into a remote invocation. The CLI picks one by name; the function is otherwise
free of `Typer`/CLI concerns so it can be unit-tested in isolation.

- `srun`     SLURM interactive (default). `ssh host "srun ... cmd"`.
- `sbatch`   SLURM queued. Renders `infra/remote/slurm.sbatch.tpl` and
             `ssh host "sbatch <generated>.sh"`. Non-blocking; the manifest
             entry gets `finished_at=None, exit_code=None` until the job
             completes - a follow-up `bin/remote reconcile` will update it.
- `ssh`      Plain `ssh host cmd`. Use this only for login-node-only work.
- `local`    No remote at all - runs the command on the local machine. Useful
             for dry-runs and integration testing.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aic2026.remote.ssh import SSHError, ssh_exec


@dataclass(frozen=True)
class LaunchResult:
    """What every launcher returns. `slurm_job_id` is None outside SLURM."""

    exit_code: int
    stdout: str
    stderr: str
    slurm_job_id: str | None = None


def launch_local(cmd: str, *, env: dict[str, str] | None = None) -> LaunchResult:
    """Run `cmd` locally. No ssh, no SLURM. Convenient for dry-runs and tests."""
    proc_env = os.environ.copy()
    proc_env.update(env or {})
    proc = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )
    return LaunchResult(
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def launch_ssh(host: str, cmd: str, *, env: dict[str, str] | None = None) -> LaunchResult:
    """Plain `ssh host cmd`. Use only for login-node-only commands."""
    try:
        proc = ssh_exec(host, cmd, env=env, check=False)
    except SSHError as exc:  # pragma: no cover - ssh_exec only raises with check=True
        return LaunchResult(exit_code=exc.returncode, stdout="", stderr=exc.stderr)
    return LaunchResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def launch_srun(
    host: str,
    cmd: str,
    *,
    env: dict[str, str] | None = None,
    srun_args: str | None = None,
) -> LaunchResult:
    """`ssh host "srun <srun_args> <cmd>"`. Blocks until the job completes.

    `srun_args` overrides the cluster default. Falls back to the env var
    `AIC2026_REMOTE_SRUN_DEFAULT_ARGS`, then to `--gpus=1 --time=02:00:00`.
    """
    args = srun_args or os.environ.get(
        "AIC2026_REMOTE_SRUN_DEFAULT_ARGS", "--gpus=1 --time=02:00:00"
    )
    full_cmd = f"srun {args} bash -c {shlex.quote(cmd)}"
    return launch_ssh(host, full_cmd, env=env)


def launch_sbatch(
    host: str,
    cmd: str,
    *,
    env: dict[str, str] | None = None,
    sbatch_tpl: Path | None = None,
    sbatch_vars: dict[str, str] | None = None,
    remote_script_path: str = "~/aic2026/last.sbatch.sh",
) -> LaunchResult:
    """Render the sbatch template, ssh-push it, `sbatch` it. Non-blocking.

    The template is a minimal Jinja2-style file (we use `str.replace` to avoid
    a Jinja2-on-remote dep; the template lives in
    [infra/remote/slurm.sbatch.tpl](../../../infra/remote/slurm.sbatch.tpl)).
    """
    tpl_path = sbatch_tpl or Path("infra/remote/slurm.sbatch.tpl")
    if not tpl_path.exists():
        return LaunchResult(
            exit_code=1,
            stdout="",
            stderr=f"sbatch template not found: {tpl_path}",
        )
    body = tpl_path.read_text(encoding="utf-8")
    rendered_vars = {**(sbatch_vars or {}), "CMD": cmd}
    for key, val in rendered_vars.items():
        body = body.replace("{{ " + key + " }}", val)
    # Write the rendered sbatch script directly to the remote via stdin so we
    # never touch local /tmp and never leak secrets through scp.
    try:
        proc = subprocess.run(
            ["ssh", host, f"cat > {remote_script_path}"],
            input=body,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return LaunchResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        submit = ssh_exec(host, f"sbatch {remote_script_path}", env=env, check=False)
    except SSHError as exc:  # pragma: no cover
        return LaunchResult(exit_code=exc.returncode, stdout="", stderr=exc.stderr)
    job_id: str | None = None
    if submit.returncode == 0:
        # "Submitted batch job 12345"
        for tok in submit.stdout.split():
            if tok.isdigit():
                job_id = tok
                break
    return LaunchResult(
        exit_code=submit.returncode,
        stdout=submit.stdout,
        stderr=submit.stderr,
        slurm_job_id=job_id,
    )
