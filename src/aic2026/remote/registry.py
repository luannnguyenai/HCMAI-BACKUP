# Implements SPEC-0022 SS 3 (job registry) and AC5.
"""Job-registration mechanism for `bin/remote run <name>`.

Each job lives under `aic2026.remote.jobs.<name>` and registers itself by
calling `@register("name")` on its entry function. Importing the `jobs`
package fires all registrations as a side effect.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aic2026.remote.context import RunContext

JobFn = Callable[[RunContext, dict[str, Any]], None]
"""A registered job: takes a RunContext + a free-form config dict; returns None.

Raise on failure. Any artifacts the job wants synced to R2 must be written
under `ctx.local_run_dir`; the runner uploads that directory in full.
"""

_REGISTRY: dict[str, JobFn] = {}


def register(name: str) -> Callable[[JobFn], JobFn]:
    """Decorator. `@register("extract-siglip")` puts the function in the table.

    Registering twice with the same name raises - it's almost always a bug
    (duplicate module import paths), and silent overrides hide it.
    """

    def deco(fn: JobFn) -> JobFn:
        if name in _REGISTRY:
            raise ValueError(
                f"job {name!r} is already registered (known jobs: {sorted(_REGISTRY)})"
            )
        _REGISTRY[name] = fn
        return fn

    return deco


def resolve(name: str) -> JobFn:
    """Look up a registered job; raise `KeyError` listing the available names."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown job {name!r}; available: {known}")
    return _REGISTRY[name]


def known_jobs() -> list[str]:
    """Sorted snapshot of the currently registered job names."""
    return sorted(_REGISTRY)
