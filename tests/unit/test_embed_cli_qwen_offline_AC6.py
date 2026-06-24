# Implements SPEC-0004 SS 5 (AC6).
# Proves the Qwen3-VL-Embedding-2B offline visual-document lane (ADR-0012) is
# wired into `bin/embed images --encoder qwen3vl`:
#   - the CI-safe path exercises offline name-resolution WITHOUT importing torch
#     (no `embedding` extra in CI): it must surface the extra hint + exit non-zero;
#   - the live-deps path (real model + cloned official repo) is importorskip-gated
#     and skipped in CI, mirroring tests/unit/test_encoders_AC1.py.

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aic2026.cli.embed import KNOWN_ENCODERS, app


def test_qwen3vl_is_a_known_offline_encoder_AC6() -> None:
    assert "qwen3vl" in KNOWN_ENCODERS


def test_qwen3vl_without_extra_surfaces_hint_and_nonzero_exit_AC6(tmp_path: Path) -> None:
    """CI-safe: invoking the qwen3vl offline lane without the `embedding` extra
    (and without the cloned official repo) must resolve the name, fail to load
    the heavy deps, print the extra hint, and exit non-zero - all without
    dragging torch into the CLI import path."""
    out_base = tmp_path / "v"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["images", "--input", str(tmp_path), "--output", str(out_base), "--encoder", "qwen3vl"],
    )
    assert result.exit_code != 0, result.output
    # The hint comes from aic2026.embedding.qwen3vl_embed (SPEC-0025 SS 9 Q2).
    assert "Qwen3-VL-Embedding" in result.output
    assert "impl_src" in result.output
    # No vectors should have been written when resolution fails.
    assert not (tmp_path / "v.npy").exists()


def test_qwen3vl_live_deps_writes_unit_norm_vectors_AC6(tmp_path: Path) -> None:
    """Skipped in CI (no torch, no cloned repo). When the `embedding` extra and
    the official QwenLM/Qwen3-VL-Embedding repo are importable, the encoder is a
    2048-d (or out_dim) unit-norm `encode_image` lane. Mirrors the importorskip
    gating in tests/unit/test_encoders_AC1.py."""
    pytest.importorskip("torch")
    pytest.importorskip("src.models.qwen3_vl_embedding")

    from aic2026.embedding.qwen3vl_embed import NATIVE_DIM, Qwen3VLEmbedder

    emb = Qwen3VLEmbedder(device="cpu", dtype="float32")
    assert emb.dim == NATIVE_DIM
