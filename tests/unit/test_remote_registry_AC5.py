# Proves SPEC-0022 AC5: job registry resolves a registered name; unknown
# names raise KeyError that lists the available jobs.

from __future__ import annotations

import pytest


def test_extract_siglip_is_registered_AC5() -> None:
    # Importing the package fires the @register("extract-siglip") side effect.
    import aic2026.remote.jobs  # noqa: F401
    from aic2026.remote.registry import known_jobs, resolve

    assert "extract-siglip" in known_jobs()
    fn = resolve("extract-siglip")
    assert fn.__name__ == "extract_siglip"
    assert fn.__module__ == "aic2026.remote.jobs.extract_siglip"


def test_unknown_job_raises_keyerror_with_available_names_AC5() -> None:
    import aic2026.remote.jobs  # noqa: F401
    from aic2026.remote.registry import known_jobs, resolve

    with pytest.raises(KeyError) as exc_info:
        resolve("definitely-not-a-job")
    msg = str(exc_info.value)
    assert "definitely-not-a-job" in msg
    # All known jobs are mentioned in the error message - this is what makes
    # the error actionable.
    for job in known_jobs():
        assert job in msg


def test_duplicate_register_raises_AC5() -> None:
    from aic2026.remote.registry import register

    @register("duplicate-test-job")  # type: ignore[arg-type]
    def fn1(ctx, cfg) -> None:
        return None

    with pytest.raises(ValueError, match="duplicate-test-job"):

        @register("duplicate-test-job")  # type: ignore[arg-type]
        def fn2(ctx, cfg) -> None:
            return None
