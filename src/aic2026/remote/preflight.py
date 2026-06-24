# Implements SPEC-0028 SS 3-4 (R2 prefix precondition check) and AC1-AC3.
"""Bank-before-consume precondition check for lease jobs (ADR-0016 rule c).

A lease box can reboot and wipe `/tmp` at any time (ADR-0016, the 2026-06-08
keyframe-loss incident). Before a job consumes an input, that input must already
exist in the durable store (R2, ADR-0011), not just on the box. This module is
the executable form of that rule: given the bucket-relative prefixes a job
requires, it verifies each exists and is non-empty.

It is read-only - it calls `R2Client.list` and nothing else. It never banks,
mutates, or deletes anything; banking stays with the `*_bank_watcher.sh` scripts
and the runner's post-run upload.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aic2026.remote.r2 import R2Client


@dataclass(frozen=True)
class PrefixStatus:
    """Result of probing one required prefix."""

    prefix: str
    present: bool
    object_count: int


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of a precondition check over a set of required prefixes."""

    ok: bool
    statuses: tuple[PrefixStatus, ...]

    def missing(self) -> tuple[str, ...]:
        """Prefixes that hold zero objects (absent or empty)."""
        return tuple(s.prefix for s in self.statuses if not s.present)


class PreflightError(RuntimeError):
    """Raised by `require_prefixes` when one or more prefixes are missing/empty.

    Carries the full `PreflightResult` on `.result` so callers can inspect every
    prefix's status, not just the message.
    """

    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        missing = ", ".join(result.missing())
        super().__init__(
            "R2 precondition failed (ADR-0016 bank-before-consume): required "
            f"prefix(es) missing or empty in R2: {missing}. Bank the input(s) "
            "before running this job; never depend on a box-local copy."
        )


def _dedupe(prefixes: Sequence[str]) -> list[str]:
    """First-seen-order de-duplication so each prefix is probed once."""
    seen: set[str] = set()
    out: list[str] = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def check_prefixes(client: R2Client, prefixes: Sequence[str]) -> PreflightResult:
    """Probe each unique prefix; report presence and object count. Never raises.

    A prefix is "present" iff `R2Client.list(prefix)` returns at least one key.
    An empty `prefixes` list is a no-op success (no R2 call), which is what makes
    the opt-in `run` guard safe to leave unset.
    """
    unique = _dedupe(prefixes)
    statuses: list[PrefixStatus] = []
    for prefix in unique:
        count = len(client.list(prefix))
        statuses.append(PrefixStatus(prefix=prefix, present=count > 0, object_count=count))
    ok = all(s.present for s in statuses)
    return PreflightResult(ok=ok, statuses=tuple(statuses))


def require_prefixes(client: R2Client, prefixes: Sequence[str]) -> PreflightResult:
    """Run `check_prefixes`; raise `PreflightError` if any prefix is missing/empty.

    Returns the `PreflightResult` on success so callers can log the counts.
    """
    result = check_prefixes(client, prefixes)
    if not result.ok:
        raise PreflightError(result)
    return result
