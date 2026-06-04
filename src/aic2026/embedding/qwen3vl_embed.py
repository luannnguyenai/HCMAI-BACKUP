# Implements SPEC-0025 SS 2.1 (Qwen3VLEmbedder).
"""Qwen3-VL-Embedding-2B unified multimodal encoder (2048-d, Apache-2.0).

The bake-off candidate (SPEC-0025): one model maps Vietnamese query text and
keyframe images into a single space (MMEB-V2 SOTA + strong visual-document
retrieval); the *same* model runs the query (online) and keyframe (offline) sides.

Integration note (SPEC-0025 SS 9 Q2, resolved on the box 2026-06-02): the official
embedding API is **not** raw ``transformers.AutoModel`` (that loads only the base
``Qwen3VLModel`` with no embedding head). It is the model's own
``Qwen3VLEmbedder`` class shipped in the QwenLM/Qwen3-VL-Embedding GitHub repo
(``src/models/qwen3_vl_embedding.py``), used with instruction-aware, list-of-dict
inputs and a ``.process()`` method. This wrapper **delegates** to that official
class (cloned on the box) so we get the correct last-token pooling + instruction
handling rather than an ad-hoc mean-pool. ``impl_src`` points at the cloned repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from aic2026.embedding.base import l2_normalize

MODEL_ID: str = "qwen3-vl-embedding-2b"
HF_REPO: str = "Qwen/Qwen3-VL-Embedding-2B"
NATIVE_DIM: int = 2048
# Query-side instruction (asymmetric: queries are instructed, image docs are not),
# per the official README example.
DEFAULT_QUERY_INSTRUCTION: str = "Retrieve images or text relevant to the user's query."

_EXTRA_HINT = (
    "Qwen3-VL-Embedding needs the `embedding` extra AND the official repo cloned: "
    "`git clone https://github.com/QwenLM/Qwen3-VL-Embedding` and pass its path as "
    "`impl_src` (SPEC-0025 SS 9 Q2)."
)


class Qwen3VLEmbedder:
    """Thin wrapper over the official ``Qwen3VLEmbedder.process()`` API.

    ``impl_src`` is the path to the cloned QwenLM/Qwen3-VL-Embedding repo (added
    to ``sys.path`` so ``src.models.qwen3_vl_embedding`` imports). ``out_dim``
    (Matryoshka) truncates + renormalises; ``None`` keeps the native 2048.
    ``impl_kwargs`` forwards extra constructor kwargs to the official class for
    on-box tuning (dtype, min/max pixels, attention impl, ...).
    """

    model_id: str = MODEL_ID

    def __init__(
        self,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        out_dim: int | None = None,
        model_name_or_path: str = HF_REPO,
        impl_src: str | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        impl_kwargs: dict[str, object] | None = None,
    ) -> None:
        if out_dim is not None and (out_dim <= 0 or out_dim > NATIVE_DIM):
            raise ValueError(f"out_dim must be in 1..{NATIVE_DIM}; got {out_dim}")
        self.dim = out_dim or NATIVE_DIM
        self._out_dim = out_dim
        self._device = device
        self._query_instruction = query_instruction

        if impl_src is not None and impl_src not in sys.path:
            sys.path.insert(0, impl_src)
        try:
            from src.models.qwen3_vl_embedding import (  # type: ignore[import-not-found]
                Qwen3VLEmbedder as _OfficialEmbedder,
            )
        except ImportError as exc:  # pragma: no cover - exercised on the box
            raise ImportError(_EXTRA_HINT) from exc

        self._impl = _OfficialEmbedder(
            model_name_or_path=model_name_or_path,
            **(impl_kwargs or {}),
        )

    # --- Embedder protocol -------------------------------------------------

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        inputs = [{"instruction": self._query_instruction, "text": t} for t in texts]
        return self._finalize(self._impl.process(inputs))

    def encode_image(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        inputs = [{"image": str(Path(p))} for p in paths]
        return self._finalize(self._impl.process(inputs))

    def _finalize(self, embeddings: object) -> np.ndarray:
        """Official ``.process()`` -> (n, native_dim); MRL-truncate + L2-normalise."""
        arr = embeddings
        if hasattr(arr, "detach"):  # torch.Tensor
            arr = arr.detach().to("cpu").float().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self._out_dim is not None:
            arr = arr[:, : self._out_dim]
        return l2_normalize(arr)
