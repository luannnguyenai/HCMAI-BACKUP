# Implements SPEC-0004 (image-embedding service).
"""Dense bi-encoders for the retrieval ensemble.

This package exposes the `Embedder` Protocol and the encoders that satisfy
it. The slice ships:

- `DummyEmbedder`: deterministic, numpy-only, CI-safe. Used by tests and by
  the rest of the package as the default until real weights are wired in.
- `SigLip2Embedder`: SigLIP-2 So400m/16@384 (1024-d). Lazy-imports torch +
  open_clip; gated behind the `[project.optional-dependencies] embedding`
  extra so CI does not install multi-GB GPU deps. See `SigLip2Embedder` for
  the install hint.

ADR-0003 splits the deployment: image-tower work (`encode_image`) runs
offline on GH200; only `encode_text` runs online on the RTX 5070.
"""

from aic2026.embedding.base import Embedder, l2_normalize
from aic2026.embedding.dummy import DummyEmbedder

__all__ = [
    "DummyEmbedder",
    "Embedder",
    "l2_normalize",
]
