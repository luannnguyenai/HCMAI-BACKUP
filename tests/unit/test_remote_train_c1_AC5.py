# Proves SPEC-0014 AC5: the `train-c1` job is registered and resolvable, and the
# registry import path stays light (heavy training deps are imported inside the
# job, not at module level) - so CI, which has no torch/transformers/datasets,
# can still import the jobs package.

from __future__ import annotations

import inspect


def test_train_c1_is_registered_AC5() -> None:
    import aic2026.remote.jobs  # noqa: F401 - fires @register side effects
    from aic2026.remote.registry import known_jobs, resolve

    assert "train-c1" in known_jobs()
    assert callable(resolve("train-c1"))


def test_train_c1_module_imports_without_heavy_deps_AC5() -> None:
    # Importing the module must not require torch/transformers/datasets/pyarrow.
    # (If it did, this import would fail in CI where none are installed.)
    import aic2026.remote.jobs.train_c1 as mod

    src = inspect.getsource(mod)
    # The heavy training imports must live inside the job function, not at top.
    top_level = "\n".join(
        line for line in src.splitlines() if line and not line.startswith((" ", "\t"))
    )
    for forbidden in (
        "import torch",
        "from aic2026.train.diacritic_bert",
        "from aic2026.train.diacritic_corpus",
        "import datasets",
    ):
        assert forbidden not in top_level, f"{forbidden!r} must be a deferred import"
