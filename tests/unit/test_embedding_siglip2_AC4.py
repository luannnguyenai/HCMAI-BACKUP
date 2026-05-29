# Proves SPEC-0004 AC4: SigLip2Embedder lazy-loads behind the `embedding`
# extra, and (when those deps are present) returns 1024-d L2-normalised
# vectors. In CI the `embedding` extra is NOT installed so the live-deps
# test is skipped via importorskip.

from __future__ import annotations

import subprocess
import sys

import pytest

from aic2026.embedding import siglip2 as siglip2_module


def test_siglip2_module_does_not_import_torch_at_import_time_AC4() -> None:
    """Lazy-import contract: `import aic2026.embedding.siglip2` must not
    drag torch into `sys.modules`.

    Runs in a fresh subprocess so it is robust against collection-order
    pollution (an `embedding`-extra env may have other tests that load
    torch). Exits 0 only if torch was NOT loaded by the import.
    """
    code = (
        "import sys, importlib;"
        " importlib.import_module('aic2026.embedding.siglip2');"
        " sys.exit(0 if 'torch' not in sys.modules else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        "importing aic2026.embedding.siglip2 loaded torch into sys.modules; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_siglip2_module_exposes_canonical_constants_AC4() -> None:
    assert siglip2_module.MODEL_ID == "siglip2-so400m-p16-384"
    assert siglip2_module.DIM == 1024
    assert siglip2_module.IMAGE_SIZE == 384


def test_siglip2_encode_text_with_real_deps_AC4() -> None:
    """Skipped in CI (no torch); runs locally when `uv sync --extra embedding`."""
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")

    from aic2026.embedding.siglip2 import SigLip2Embedder

    emb = SigLip2Embedder(device="cpu", dtype="float32")
    out = emb.encode_text(["hello world"])
    assert out.shape == (1, 1024)
    norms = (out**2).sum(axis=1) ** 0.5
    assert abs(float(norms[0]) - 1.0) < 1e-3
