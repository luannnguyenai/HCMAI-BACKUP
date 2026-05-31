# Proves SPEC-0014 AC9: the `eval-c1` remote job is registered and the registry
# import path stays light (heavy deps are imported inside the job function, not
# at module top-level), so CI - which has neither torch nor transformers - can
# still import the jobs package.

from __future__ import annotations

import inspect


def test_eval_c1_is_registered_AC9() -> None:
    import aic2026.remote.jobs  # noqa: F401 - fires @register side effects
    from aic2026.remote.registry import known_jobs, resolve

    assert "eval-c1" in known_jobs()
    assert callable(resolve("eval-c1"))


def test_eval_c1_module_imports_without_heavy_deps_AC9() -> None:
    import aic2026.remote.jobs.eval_c1 as mod

    src = inspect.getsource(mod)
    top_level = "\n".join(
        line for line in src.splitlines() if line and not line.startswith((" ", "\t"))
    )
    for forbidden in (
        "import torch",
        "from aic2026.eval.diacritic_robustness",
        "from aic2026.eval.retrievers",
        "from aic2026.train.diacritic_bert",
        "import transformers",
    ):
        assert forbidden not in top_level, f"{forbidden!r} must be a deferred import"
