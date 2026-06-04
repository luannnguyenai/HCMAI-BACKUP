# Implements SPEC-0025 SS 2.1 (ProvidedClipEmbedder).
"""The organisers' provided CLIP ViT-B/32 baseline lane (512-d).

This is the "weak CLIP" the organisers ship pre-computed (research-note 06 SS 2.3,
research-note 07 SS 4 -- `clip-features-32`). The bake-off includes it as the
fair baseline our own encoders must beat. Two halves:

  * ``encode_image`` does **not** run a model -- it looks up the organisers'
    pre-extracted vector by frame id (the image tower already ran offline).
  * ``encode_text`` runs the matching ``openai/clip-vit-base-patch32`` text
    tower so query vectors land in the same 512-d space as the provided image
    features.

The provided-feature *layout* is verify-on-box (SPEC-0025 SS 9 Q3): `from_dir`
handles the two common AIC layouts and raises a clear error otherwise so the
real format surfaces. `key_fn` maps a keyframe path to the id convention used by
the feature store (default ``"<videoDir>_<frameStem>"``, which is unique across
videos -- a bare stem like ``001`` is not).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aic2026.embedding.base import l2_normalize

logger = logging.getLogger(__name__)

MODEL_ID: str = "provided-clip-vit-b32"
DIM: int = 512
TEXT_HF_REPO: str = "openai/clip-vit-base-patch32"

_EXTRA_HINT = (
    "Provided-CLIP text tower deps are not installed. Install the `embedding` "
    "extra: `uv sync --extra embedding`. See SPEC-0025 SS 7."
)


def _default_key(path: Path) -> str:
    """Composite, video-unique frame id: ``<parentDirName>_<stem>``."""
    return f"{path.parent.name}_{path.stem}"


class ProvidedClipEmbedder:
    """Pre-extracted CLIP image vectors + the CLIP-B/32 text tower for queries."""

    model_id: str = MODEL_ID
    dim: int = DIM

    def __init__(
        self,
        features: dict[str, np.ndarray],
        *,
        device: str = "cpu",
        strict: bool = True,
        key_fn: Callable[[Path], str] = _default_key,
    ) -> None:
        if not features:
            raise ValueError("features mapping is empty")
        any_vec = next(iter(features.values()))
        self.dim = int(np.asarray(any_vec).reshape(-1).shape[0])
        self._features = {
            k: l2_normalize(np.asarray(v, np.float32).reshape(1, -1))[0]
            for k, v in features.items()
        }
        self._device = device
        self._strict = strict
        self._key_fn = key_fn
        self._text_model = None  # lazy
        self._text_proc = None

    # --- feature-store loaders --------------------------------------------

    @classmethod
    def from_dir(cls, features_dir: Path, **kwargs: object) -> ProvidedClipEmbedder:
        """Best-effort load of the provided feature store (SPEC-0025 Q3).

        Searched recursively (the AIC zip nests the .npy under a
        ``clip-features-32/`` subdir). Three layouts, in order:

        * **A** one ``*.npy`` matrix + a row-aligned id list (``*.json`` array,
          ``*.jsonl`` of {id|frame_id}, or ``*.txt`` one-per-line).
        * **C** per-**video** matrices ``<video>.npy`` of shape ``(n_frames, d)``
          -- the AIC2025 layout (verified on box: ``L25_V011.npy`` -> (318, 512)).
          Keys are ``f"{video}_{i+1:03d}"`` so row ``i`` aligns with keyframe
          ``<video>/<i+1:03d>.jpg`` (the organiser keyframes are 1-based 3-digit;
          assumption recorded in SPEC-0025 Q3).
        * **B** per-**frame** ``<id>.npy`` single vectors.

        Raises with the listing if none match, so the real layout surfaces.
        """
        features_dir = Path(features_dir)
        npys = sorted(features_dir.rglob("*.npy"))
        mapping = (
            _load_layout_a(features_dir, npys) or _load_per_video(npys) or _load_layout_b(npys)
        )
        if mapping is None:
            listing = ", ".join(p.name for p in npys[:10]) or "(no .npy found)"
            raise ValueError(
                f"could not parse provided-CLIP features under {features_dir} "
                f"(tried matrix+ids, per-video matrices, per-id .npy). Entries: {listing}"
            )
        return cls(mapping, **kwargs)  # type: ignore[arg-type]

    # --- Embedder protocol -------------------------------------------------

    def encode_image(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = np.zeros((len(paths), self.dim), dtype=np.float32)
        missing = 0
        for i, p in enumerate(paths):
            key = self._key_fn(Path(p))
            vec = self._features.get(key)
            if vec is None:
                if self._strict:
                    raise KeyError(f"no provided-CLIP feature for frame id {key!r}")
                missing += 1
                continue
            rows[i] = vec
        if missing:
            logger.warning(
                "provided-CLIP: %d/%d frames had no feature (zero-filled)", missing, len(paths)
            )
        return rows

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_text_tower()
        import torch  # type: ignore[import-not-found]

        inputs = self._text_proc(  # type: ignore[misc]
            text=list(texts), padding=True, truncation=True, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            feats = self._text_model.get_text_features(**inputs)  # type: ignore[union-attr]
        # transformers 5.x may return a ModelOutput rather than a bare tensor.
        if not hasattr(feats, "detach"):
            feats = (
                getattr(feats, "text_embeds", None)
                if getattr(feats, "text_embeds", None) is not None
                else getattr(feats, "pooler_output", feats[0])
            )
        arr = feats.detach().to("cpu", dtype=torch.float32).numpy()
        return l2_normalize(arr)

    def _ensure_text_tower(self) -> None:
        if self._text_model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_EXTRA_HINT) from exc
        self._text_model = CLIPModel.from_pretrained(TEXT_HF_REPO).to(self._device).eval()
        self._text_proc = CLIPProcessor.from_pretrained(TEXT_HF_REPO)


def _load_layout_a(features_dir: Path, npys: list[Path]) -> dict[str, np.ndarray] | None:
    """One matrix .npy + a row-aligned id list."""
    if len(npys) != 1:
        return None
    matrix = np.load(npys[0])
    if matrix.ndim != 2:
        return None
    ids = _load_ids(features_dir, n=matrix.shape[0])
    if ids is None or len(ids) != matrix.shape[0]:
        return None
    return {fid: matrix[i] for i, fid in enumerate(ids)}


def _load_per_video(npys: list[Path]) -> dict[str, np.ndarray] | None:
    """Per-video matrices ``<video>.npy`` (n_frames, d) -> keys ``<video>_<NNN>``."""
    if len(npys) < 2:
        return None
    first = np.load(npys[0])
    if first.ndim != 2 or first.shape[0] < 2:
        return None  # not per-video matrices; let layout B try
    out: dict[str, np.ndarray] = {}
    for p in npys:
        mat = np.load(p)
        if mat.ndim != 2:
            continue
        for i in range(mat.shape[0]):
            out[f"{p.stem}_{i + 1:03d}"] = mat[i]
    return out or None


def _load_ids(features_dir: Path, *, n: int) -> list[str] | None:
    for cand in sorted(features_dir.glob("*.json")) + sorted(features_dir.glob("*.jsonl")):
        try:
            text = cand.read_text(encoding="utf-8")
        except OSError:
            continue
        if cand.suffix == ".json":
            obj = json.loads(text)
            ids = obj if isinstance(obj, list) else None
        else:
            ids = [
                json.loads(ln).get("frame_id") or json.loads(ln).get("id")
                for ln in text.splitlines()
                if ln.strip()
            ]
        if ids and len(ids) == n:
            return [str(x) for x in ids]
    for cand in sorted(features_dir.glob("*.txt")):
        lines = [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) == n:
            return lines
    return None


def _load_layout_b(npys: list[Path]) -> dict[str, np.ndarray] | None:
    """Per-frame ``<id>.npy`` single-vector files."""
    if len(npys) < 2:
        return None
    out: dict[str, np.ndarray] = {}
    for p in npys:
        out[p.stem] = np.load(p).reshape(-1)
    return out or None
