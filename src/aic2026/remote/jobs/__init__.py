# Implements SPEC-0022 SS 3 (jobs package).
"""Importing this package fires the `@register("...")` side effects.

When the runner's CLI starts up it does `import aic2026.remote.jobs` exactly
once; every job module listed below is then in the registry and resolvable
via `aic2026.remote.registry.resolve(name)`.
"""

from aic2026.remote.jobs import (
    cache_env,  # noqa: F401  (import for side effect)
    cache_weights,  # noqa: F401  (import for side effect)
    extract_siglip,  # noqa: F401  (import for side effect)
    train_c1,  # noqa: F401  (import for side effect)
)

__all__: list[str] = []
