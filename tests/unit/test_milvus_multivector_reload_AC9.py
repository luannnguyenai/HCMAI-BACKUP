# Proves SPEC-0006 AC9: a multi-vector collection persisted by one process and
# reopened COLD by a fresh process (the ingest -> serve pattern) loads and
# searches every dense field at its correct dim. This is the regression guard
# for the milvus-lite 3.0 multi-vector reload cross-wiring (SPEC-0006 SS 12):
# the dependency pin (pymilvus/milvus-lite < 3) must keep this green.
#
# The bug only surfaces on a truly fresh interpreter re-deserialising the
# on-disk segments, so this drives the real store via two sequential
# subprocesses (build, then load) rather than an in-process reopen.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pymilvus", reason="pymilvus not installed (index extra)")
pytest.importorskip("milvus_lite", reason="milvus-lite backend not installed")

_HELPER = Path(__file__).with_name("_milvus_reload_helper.py")


def _run(*args: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(_HELPER), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not out:  # pragma: no cover - only on an unexpected crash
        raise AssertionError(
            f"helper {args} produced no output (rc={proc.returncode}); stderr:\n{proc.stderr}"
        )
    return out


@pytest.mark.integration
def test_multi_vector_collection_loads_per_field_dim_AC9(tmp_path: Path) -> None:
    uri = str(tmp_path / "reload.db")
    src_root = str(tmp_path / "src")

    build = _run("build", uri, src_root)
    if build[0].startswith("SKIP:"):
        pytest.skip(f"Milvus Lite could not initialise in this environment: {build[0]}")
    assert build[-1] == "BUILD_OK", f"unexpected build output: {build}"

    load = _run("load", uri)
    if load[0].startswith("SKIP:"):
        pytest.skip(f"Milvus Lite could not initialise in this environment: {load[0]}")

    # load_collection must succeed in the fresh process (the milvus-lite 3.0
    # regression raised "loaded index dim 1024 != expected dim 1152" here).
    assert "LOAD_OK" in load, f"cold reload failed: {load}"

    # Every dense field must search at its own declared dim, not the first
    # field's dim (the cross-wiring symptom).
    field_lines = [ln for ln in load if ln.count(":") >= 2 and not ln.startswith("LOAD")]
    assert field_lines, f"no per-field results: {load}"
    for line in field_lines:
        assert line.endswith(":OK"), f"per-field reload/search failed: {line!r} (all: {load})"
    # All three floor lanes covered.
    assert {ln.split(":", 1)[0] for ln in field_lines} == {"siglip2", "metaclip2", "qwen3vl"}
