# Implements SPEC-0004 SS 3-4 (DummyEmbedder).
"""Deterministic, numpy-only encoder used by tests and CI.

Why this exists: the real backbones live in an optional dependency extra
(`embedding`) that CI deliberately does not install (multi-GB torch +
CUDA). Tests for the harness, the extraction CLI, and downstream specs
(SPEC-0006 Milvus, SPEC-0015 fusion) all need *some* encoder to call;
`DummyEmbedder` is that stand-in.

The vectors are produced by seeding a NumPy `Generator` with
`sha256(f"{model_id}|{input_item}")` and drawing a standard-normal vector,
then L2-normalising. This gives:

- Determinism: same `(model_id, dim, input)` -> bit-identical vector.
- Input-distinctness: distinct inputs produce distinct seeds.
- No collisions with real-model outputs: `model_id` carries a `dummy-`
  prefix and is not a valid SigLIP-2 / Meta CLIP 2 identifier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from aic2026.embedding.base import l2_normalize

DEFAULT_DIM: int = 64


def _seed_from(model_id: str, item: str) -> int:
    """64-bit deterministic seed from `sha256("model_id|item")`."""
    digest = hashlib.sha256(f"{model_id}|{item}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _read_bytes_safe(path: Path) -> bytes:
    """Read file bytes for hashing; returns `b""` when the path is missing.

    The dummy encoder does *not* decode the image - it only hashes the raw
    bytes. That keeps CI free of `pillow`/decoder deps and still gives a
    deterministic vector per file content.
    """
    try:
        return path.read_bytes()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return b""


class DummyEmbedder:
    """Deterministic numpy encoder; identifies itself as `dummy-<dim>`."""

    def __init__(self, *, dim: int = DEFAULT_DIM, model_id: str | None = None) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        self.dim = dim
        self.model_id = model_id or f"dummy-{dim}"

    # --- core helpers ------------------------------------------------------

    def _encode_one(self, item: str) -> np.ndarray:
        rng = np.random.default_rng(_seed_from(self.model_id, item))
        return rng.standard_normal(self.dim, dtype=np.float32)

    def _encode_batch(self, items: list[str]) -> np.ndarray:
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.stack([self._encode_one(it) for it in items], axis=0)
        return l2_normalize(matrix)

    # --- Embedder protocol -------------------------------------------------

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return self._encode_batch(list(texts))

    def encode_image(self, paths: list[Path]) -> np.ndarray:
        # Hash the file bytes (or empty bytes for missing files) so two
        # different image files yield different vectors without decoding
        # the pixels. The path string is *not* used as the seed source -
        # we want content-keyed determinism, not path-keyed.
        items: list[str] = []
        for p in paths:
            raw = _read_bytes_safe(Path(p))
            digest = hashlib.sha256(raw).hexdigest()
            items.append(f"image:{digest}")
        return self._encode_batch(items)
