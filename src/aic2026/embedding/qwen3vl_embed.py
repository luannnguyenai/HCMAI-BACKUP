# Implements SPEC-0025 SS 2.1 (Qwen3VLEmbedder).
"""Qwen3-VL-Embedding-2B unified multimodal encoder (2048-d, Apache-2.0).

The bake-off candidate (SPEC-0025): one model maps both Vietnamese query text
and keyframe images into a single space, with MMEB-V2 SOTA + strong visual-
document retrieval. Unlike the CLIP-style dual encoders, the *same* model runs
the query (online) and the keyframe (offline) sides.

Integration risk (SPEC-0025 SS 9 Q2): the exact `transformers` embedding API
for Qwen3-VL-Embedding is version-sensitive and uses `trust_remote_code`. The
model-specific "forward -> pooled embedding" step is isolated in `_embed_*` so
it is the single place to adjust on the box against the HF model card; the rest
(L2-norm, MRL truncation, the Embedder protocol) is stable. Heavy deps are
lazy-imported (CI-safe, `embedding`-extra gated).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aic2026.embedding.base import l2_normalize

MODEL_ID: str = "qwen3-vl-embedding-2b"
HF_REPO: str = "Qwen/Qwen3-VL-Embedding-2B"
NATIVE_DIM: int = 2048

_EXTRA_HINT = (
    "Qwen3-VL-Embedding deps are not installed. Install the `embedding` extra: "
    "`uv sync --extra embedding` (transformers must be new enough for "
    "Qwen3-VL-Embedding; see SPEC-0025 SS 7). "
)


class Qwen3VLEmbedder:
    """Qwen3-VL-Embedding-2B; requires the `embedding` extra + a recent transformers.

    ``out_dim`` (Matryoshka) truncates + renormalises to a smaller width; ``None``
    keeps the native 2048. ``load_in_4bit`` requests a bitsandbytes 4-bit load
    (for the INT4 deployability measurement, SPEC-0025 SS 4).
    """

    model_id: str = MODEL_ID

    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: str = "float16",
        out_dim: int | None = None,
        load_in_4bit: bool = False,
        hf_repo: str = HF_REPO,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoProcessor  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise ImportError(_EXTRA_HINT) from exc

        if out_dim is not None and (out_dim <= 0 or out_dim > NATIVE_DIM):
            raise ValueError(f"out_dim must be in 1..{NATIVE_DIM}; got {out_dim}")

        self._torch = torch
        self._device = device
        self._dtype = getattr(torch, dtype)
        self.dim = out_dim or NATIVE_DIM
        self._out_dim = out_dim

        load_kwargs: dict[str, object] = {"trust_remote_code": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = self._dtype

        self._processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
        model = AutoModel.from_pretrained(hf_repo, **load_kwargs)
        if not load_in_4bit:
            model = model.to(device)
        model.eval()
        self._model = model

    # --- model-specific embedding extraction (the spot to adjust on the box) ---

    def _pool(self, outputs: object, attention_mask: object) -> object:
        """Pool a forward pass to a single vector per item.

        Prefers an explicit embedding the model exposes; else mean-pools the
        last hidden state over the attention mask. Kept defensive because the
        Qwen3-VL-Embedding head API is version-sensitive (SPEC-0025 Q2).
        """
        emb = getattr(outputs, "embeddings", None)
        if emb is None:
            emb = getattr(outputs, "pooler_output", None)
        if emb is not None:
            return emb
        hidden = outputs.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _finalize(self, emb: object) -> np.ndarray:
        torch = self._torch
        arr = emb.detach().to("cpu", dtype=torch.float32).numpy()
        if self._out_dim is not None:
            arr = arr[:, : self._out_dim]
        return l2_normalize(arr)

    # --- Embedder protocol -------------------------------------------------

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        inputs = self._processor(text=list(texts), padding=True, return_tensors="pt").to(
            self._device
        )
        with torch.inference_mode():
            outputs = self._model(**inputs)
        return self._finalize(self._pool(outputs, inputs["attention_mask"]))

    def encode_image(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_EXTRA_HINT) from exc

        torch = self._torch
        images = [Image.open(Path(p)).convert("RGB") for p in paths]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # Image branch may not carry an attention mask; pass a ones-like fallback.
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(len(images), 1, device=self._device)
        return self._finalize(self._pool(outputs, mask))
