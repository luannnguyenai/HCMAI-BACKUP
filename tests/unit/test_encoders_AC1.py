# Proves SPEC-0025 AC1: the three new encoders lazy-import torch (no torch in
# sys.modules at module-import time) and expose the declared protocol constants.
# Live-deps paths (model downloads) are validated on the GPU box, not CI.

from __future__ import annotations

import subprocess
import sys

import pytest

from aic2026.embedding import metaclip2, provided_clip, qwen3vl_embed

_MODULES = [
    "aic2026.embedding.metaclip2",
    "aic2026.embedding.qwen3vl_embed",
    "aic2026.embedding.provided_clip",
]


@pytest.mark.parametrize("module", _MODULES)
def test_encoder_module_does_not_import_torch_at_import_time_AC1(module: str) -> None:
    """Importing the encoder module must not drag torch into sys.modules."""
    code = (
        "import sys, importlib;"
        f" importlib.import_module('{module}');"
        " sys.exit(0 if 'torch' not in sys.modules else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, (
        f"importing {module} loaded torch; stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_encoder_constants_AC1() -> None:
    assert metaclip2.MODEL_ID == "metaclip2-worldwide-huge-h14"
    assert metaclip2.DIM == 1024
    assert qwen3vl_embed.MODEL_ID == "qwen3-vl-embedding-2b"
    assert qwen3vl_embed.NATIVE_DIM == 2048
    assert provided_clip.MODEL_ID == "provided-clip-vit-b32"
    assert provided_clip.DIM == 512
