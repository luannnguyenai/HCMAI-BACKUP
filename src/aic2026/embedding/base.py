# Implements SPEC-0004 SS 3 (Embedder Protocol + l2_normalize).
"""The bi-encoder Protocol and the row-wise L2-normalisation helper.

All concrete encoders return `float32` `(n, dim)` arrays whose rows have
unit L2 norm (cosine similarity reduces to dot product). The Protocol is
sized for all three SS 5.3 encoders (SigLIP-2 1024-d, Meta CLIP 2 1024-d,
InternVideo2 768-d); each concrete class declares its own `dim`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """A dense bi-encoder for image-text retrieval.

    `model_id` is a stable, repo-controlled identifier (used to seed the
    deterministic dummy encoder and to disambiguate cached vectors).
    `dim` is the output dimensionality.

    `encode_image` is offline-only on the deployment hardware split
    (ADR-0003); `encode_text` is the only method called on the online
    RTX 5070 hot path.
    """

    model_id: str
    dim: int

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Return `float32` `(len(texts), dim)`; rows L2-normalised."""

    def encode_image(self, paths: list[Path]) -> np.ndarray:
        """Return `float32` `(len(paths), dim)`; rows L2-normalised.

        Offline-only per ADR-0003: vectors are produced on the GH200 and
        pre-indexed; this method is not part of the online (RTX 5070) hot
        path.
        """


def l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation. Returns `float32`, shape unchanged.

    Zero-norm rows are left at their original direction (all-zero) rather
    than NaN-ed, by clamping the divisor with `eps`.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {x.shape}")
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32, copy=False)
