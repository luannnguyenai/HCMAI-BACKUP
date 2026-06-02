# Implements SPEC-0025 SS 2.1 (MetaClip2Embedder).
"""Meta CLIP 2 ViT-H/14 bi-encoder (1024-d) -- our Vietnamese-specialist lane.

Meta CLIP 2 (research-note 03 A.2) is the reigning multilingual cross-modal
encoder (XM3600 I->T 64.3), the most relevant rival to Qwen3-VL-Embedding for
Vietnamese queries. Loaded through ``open_clip``'s HF-hub path so the exact
checkpoint can be corrected on the box without a code change (SPEC-0025 SS 9 Q*).

Heavy deps (``torch``, ``open_clip``, ``pillow``) are imported lazily inside the
constructor / ``encode_*`` so the package imports cleanly in CI without the
``embedding`` extra (same contract as ``siglip2.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aic2026.embedding.base import l2_normalize

MODEL_ID: str = "metaclip2-worldwide-huge-h14"
DIM: int = 1024
# open_clip HF-hub spec; override via the constructor if the registry name
# differs on the box (verify against the open_clip / HF model card).
DEFAULT_OPENCLIP_REF: str = "hf-hub:facebook/metaclip-2-worldwide-huge-quickgelu"

_EXTRA_HINT = (
    "Meta CLIP 2 deps are not installed. Install the `embedding` extra: "
    "`uv sync --extra embedding`. See SPEC-0025 SS 7."
)


class MetaClip2Embedder:
    """Meta CLIP 2 ViT-H/14 encoder; requires the `embedding` extra."""

    model_id: str = MODEL_ID
    dim: int = DIM

    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: str = "float16",
        openclip_ref: str = DEFAULT_OPENCLIP_REF,
    ) -> None:
        try:
            import open_clip  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise ImportError(_EXTRA_HINT) from exc

        torch_dtype = getattr(torch, dtype)
        self._torch = torch
        self._device = device
        self._dtype = torch_dtype

        model, _, preprocess = open_clip.create_model_and_transforms(openclip_ref)
        model = model.to(device=device, dtype=torch_dtype)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(openclip_ref)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        tokens = self._tokenizer(list(texts)).to(self._device)
        with torch.inference_mode():
            features = self._model.encode_text(tokens)
        arr = features.detach().to("cpu", dtype=torch.float32).numpy()
        return l2_normalize(arr)

    def encode_image(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_EXTRA_HINT) from exc

        torch = self._torch
        tensors = []
        for p in paths:
            with Image.open(Path(p)) as im:
                tensors.append(self._preprocess(im.convert("RGB")))
        batch = torch.stack(tensors).to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            features = self._model.encode_image(batch)
        arr = features.detach().to("cpu", dtype=torch.float32).numpy()
        return l2_normalize(arr)
