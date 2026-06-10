---
id: SPEC-0028
title: R2 prefix precondition check for lease jobs
status: Implementing
owner: unassigned
created: 2026-06-08
updated: 2026-06-08
implements_proposal: docs/proposals/05-evaluation-harness.md SS 5
related_adrs:
  - ADR-0011
  - ADR-0016
depends_on:
  - SPEC-0022
---

# SPEC-0028 - R2 prefix precondition check for lease jobs

> A small, opt-in guard for the `bin/remote` runner that verifies a job's
> required R2 prefixes exist and are non-empty before the job starts, failing
> fast with a clear message otherwise. It is the executable form of
> [ADR-0016](../adr/ADR-0016-data-durability-three-tier-banking.md) rule (c):
> bank-before-consume, enforced at the start of a lease job.

## 1. Context

The 2026-06-08 keyframe-loss incident ([ADR-0016](../adr/ADR-0016-data-durability-three-tier-banking.md))
showed that a lease job can silently depend on a box-local input that was never
banked to R2. When the box rebooted, the input was gone. ADR-0016 rule (c)
requires a precondition check: before a job runs, the prefixes it consumes must
already exist in R2 and be non-empty. This spec defines that check and wires it
into the [SPEC-0022](SPEC-0022-remote-gpu-runner.md) runner as an opt-in guard.
The check is a precondition only - it does not bank anything, mutate R2, or
change how jobs upload their outputs.

## 2. Scope

### 2.1 In scope

- A pure function over `R2Client` that, given a list of bucket-relative prefixes,
  reports for each whether it exists and how many objects it holds.
- A `require_*` variant that raises a typed error naming the missing/empty
  prefixes, for fail-fast call sites.
- A `bin/remote preflight` subcommand that runs the check from the CLI and maps
  the outcome to a process exit code.
- An opt-in `--require-prefix` flag on `bin/remote run` that runs the check
  before dispatch when supplied; absent the flag, `run` behaves exactly as
  before (no R2 precondition call).

### 2.2 Out of scope

- **Banking** any artifact (Tier 1 upload, Tier 2 incremental sync). The
  `*_bank_watcher.sh` scripts and `run`'s post-run upload own that.
- **Content validation** beyond "the prefix has >= 1 object". Object count,
  schema, or checksum validation is not in scope.
- **A persisted R2 inventory document.** The "inventory" of ADR-0016 rule (c) is
  expressed as the per-job list of required prefixes passed to the check; a
  committed manifest file is a possible follow-up, not this spec.
- **Changing existing flows.** The guard is opt-in; default behaviour is unchanged.

## 3. API contract / interface

```python
# aic2026/remote/preflight.py
from collections.abc import Sequence
from dataclasses import dataclass

from aic2026.remote.r2 import R2Client

@dataclass(frozen=True)
class PrefixStatus:
    prefix: str
    present: bool          # True iff object_count > 0
    object_count: int

@dataclass(frozen=True)
class PreflightResult:
    ok: bool                              # True iff every checked prefix is present
    statuses: tuple[PrefixStatus, ...]
    def missing(self) -> tuple[str, ...]: ...   # prefixes with object_count == 0

class PreflightError(RuntimeError):
    """Raised by require_prefixes when one or more prefixes are missing/empty."""
    result: "PreflightResult"

def check_prefixes(client: R2Client, prefixes: Sequence[str]) -> PreflightResult:
    """Probe each prefix via R2Client.list; never raises on a missing prefix."""
    ...

def require_prefixes(client: R2Client, prefixes: Sequence[str]) -> PreflightResult:
    """Run check_prefixes; raise PreflightError if result.ok is False."""
    ...
```

```
bin/remote preflight --require <prefix> [--require <prefix> ...]
                     # exit 0 if all present; exit 5 (precondition) if any missing;
                     # exit 2 (usage) if no --require given

bin/remote run <job> [--require-prefix <prefix> ...] [--launcher ...] [--dry-run]
                     # when --require-prefix is supplied (>=1), require_prefixes runs
                     # before dispatch; a failure exits 5 and no job is launched.
                     # --dry-run lists the required prefixes and performs no R2 call.
```

## 4. Behaviour

- **All present**: every prefix returns >= 1 object from `R2Client.list`;
  `check_prefixes` returns `ok=True` with per-prefix counts; `require_prefixes`
  returns the same result without raising.
- **Missing / empty prefix**: a prefix returning 0 objects yields
  `PrefixStatus(present=False, object_count=0)` and `ok=False`;
  `require_prefixes` raises `PreflightError` whose message names every missing
  prefix and carries the full `PreflightResult`.
- **Empty input list**: `check_prefixes(client, [])` returns `ok=True` with no
  statuses (a no-op success). This is what makes the `run` guard safe to leave
  unset - an empty required list never calls R2 and never fails.
- **Duplicate prefixes**: de-duplicated, preserving first-seen order; each unique
  prefix is probed once.
- **CLI `preflight`**: all-present -> exit 0; any-absent -> exit 5 with a message
  naming the absent prefix(es); no `--require` given -> exit 2 (usage). Prints a
  one-line-per-prefix status report.
- **CLI `run --require-prefix`**: when one or more are supplied and `--dry-run`
  is not set, `require_prefixes` runs before the job is dispatched; a failure
  exits 5 and nothing is launched, uploaded, or appended to the ledger. With
  `--dry-run`, the required prefixes are listed in the plan and no R2 call is
  made. With no `--require-prefix`, `run` is byte-for-byte its current behaviour.

## 5. Acceptance criteria

- **AC1**: `check_prefixes` over a moto-backed bucket where all requested
  prefixes hold >= 1 object returns `ok=True`, one `PrefixStatus` per unique
  prefix with the correct `object_count`, and `missing() == ()`. Verified in
  `tests/unit/test_remote_preflight_AC1.py`.
- **AC2**: when at least one prefix is absent/empty, `check_prefixes` returns
  `ok=False` with that prefix marked `present=False`, and `require_prefixes`
  raises `PreflightError` whose message names the missing prefix and whose
  `.result` is the same `PreflightResult`. Verified in
  `tests/unit/test_remote_preflight_AC2.py`.
- **AC3**: `check_prefixes(client, [])` returns `ok=True` with empty `statuses`
  and makes no `R2Client.list` call (no-op success), and duplicate prefixes are
  probed once. Verified in `tests/unit/test_remote_preflight_AC3.py`
  (a counting fake client; no network).
- **AC4**: `bin/remote preflight` exits 0 when all `--require` prefixes are
  present, exits 5 and names the absent prefix when one is missing, and exits 2
  when no `--require` is given. Verified in
  `tests/unit/test_remote_preflight_cli_AC4.py` (Typer `CliRunner` + a patched
  `R2Client`).
- **AC5**: `bin/remote run <job> --dry-run --require-prefix <p>` exits 0, lists
  `<p>` in the printed plan, and performs no R2 precondition call (side-effect
  free); `bin/remote run <job>` with no `--require-prefix` performs no
  precondition call. Verified in `tests/unit/test_remote_preflight_run_AC5.py`.

## 6. Non-functional requirements

- **Latency**: the check is one `list_objects_v2` page probe per unique prefix
  (it can stop at the first object); negligible against job runtime. No NFR budget
  beyond "does not add measurable startup cost for a handful of prefixes".
- **Safety**: read-only against R2 (`list` only); never writes, deletes, or banks.
- **Compatibility**: Python 3.11+; reuses the existing `boto3`/`R2Client` surface
  (no new dependency). Tested with the existing `moto[server]` dev dependency.

## 7. Dependencies

- **Internal**: SPEC-0022 (`R2Client`, the `bin/remote` CLI this extends).
- **External**: none new (`boto3` already core; `moto[server]` already dev).
- **Data**: none; the check is parameterised by the caller's prefix list.

## 8. Test plan

- **Unit tests** (`tests/unit/`, moto + CliRunner, CPU/offline):
  - `test_remote_preflight_AC1.py`, `test_remote_preflight_AC2.py`,
    `test_remote_preflight_AC3.py`, `test_remote_preflight_cli_AC4.py`,
    `test_remote_preflight_run_AC5.py`.
- **Manual (lease)**: before a keyframe-consuming job, run
  `bin/remote preflight --require <keyframe-prefix>` and confirm it gates a
  not-yet-banked input (ADR-0016 rule c).

## 9. Open questions

- **Q1**: A committed R2 inventory document (per-job required-prefix lists in
  version control) vs the current call-site list. Deferred; the call-site list
  is sufficient to enforce rule (c) now.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-06-08 | implementer (AI, user-directed) | Created; Draft -> Approved -> Implementing in one pass per CONTRIBUTING. Defines the ADR-0016 rule (c) precondition check: `check_prefixes` / `require_prefixes` over `R2Client`, a `bin/remote preflight` subcommand, and an opt-in `--require-prefix` guard on `bin/remote run`. Read-only; existing flows unchanged by default. |
