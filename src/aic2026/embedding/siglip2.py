# Implements SPEC-0004 SS 3-4 (SigLip2Embedder).
"""SigLIP-2 So400m/16@384 bi-encoder (1024-d).

Heavy deps (`torch`, `open_clip`, `pillow`) are **not** imported at module
import time. They are imported lazily inside the constructor and the
`encode_*` methods so that:

- The package imports cleanly in CI (where the `embedding` extra is not
  installed).
- A module-level `import aic2026.embedding` does not pay multi-second torch
  startup cost.

Install the heavy deps locally with `uv sync --extra embedding` before
constructing this class. See SPEC-0004 SS 7 and ADR-0003 SS Decision SS 1
for the deployment split (image tower offline only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aic2026.embedding.base import l2_normalize

# Canonical SigLIP-2 So400m/16@384 reference.
# DIM is the So400m embedding width = 1152, VERIFIED on real hardware
# (8x H200 lease, 2026-05-30): `encode_text(["xin chao"]).shape == (1, 1152)`.
# Note: proposal-01 SS 5.3 originally claimed 1024-d; that was an unverified
# number and is wrong for So400m. Do not "correct" this back to 1024.
MODEL_ID: str = "siglip2-so400m-p16-384"
DIM: int = 1152
IMAGE_SIZE: int = 384

_EXTRA_HINT = (
    "SigLIP-2 deps are not installed in this environment. Install the "
    "`embedding` extra: `uv sync --extra embedding`. See SPEC-0004 SS 7."
)


class SigLip2Embedder:
    """Real SigLIP-2 encoder; requires the `embedding` extra.

    Attributes:
        model_id: "siglip2-so400m-p16-384"
        dim: 1024
    """

    model_id: str = MODEL_ID
    dim: int = DIM

    def __init__(self, *, device: str = "cpu", dtype: str = "float16") -> None:
        try:
            import open_clip  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise ImportError(_EXTRA_HINT) from exc

        # Resolve dtype lazily so we never reference `torch.float16` until
        # torch is imported.
        torch_dtype = getattr(torch, dtype)
        self._torch = torch
        self._device = device
        self._dtype = torch_dtype

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-SO400M-16-SigLIP2-384",
            pretrained="webli",
        )
        model = model.to(device=device, dtype=torch_dtype)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-384")

    # --- Embedder protocol -------------------------------------------------

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
