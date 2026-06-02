# Implements SPEC-0025 SS 3-4 (encoder bake-off harness).
"""Offline encoder bake-off: qualitative side-by-side + deployability.

No ground truth (SPEC-0025 SS 1), so this is a directional screen:

  * :func:`run_qualitative` indexes a sampled set of keyframes with each
    encoder (text-query -> image-doc), takes top-k per query, and writes an
    HTML contact sheet (one section per query, one row of top-k thumbnails per
    encoder) for human judgment.
  * :func:`measure_deployability` times ``encode_text`` (p50/p95) and, on CUDA,
    records the peak VRAM of an ``encode_text`` forward -- the portable proxy
    for the RTX 5070 online-fit gate (ADR-0003). On CPU/CI the VRAM fields are
    ``None``.

Pure-numpy retrieval (L2-normalised dot product, argsort top-k) -- no Milvus.
Torch is only touched if the passed encoder is torch-backed; the harness itself
imports cleanly on CPU and is exercised in CI with ``DummyEmbedder``.
"""

from __future__ import annotations

import dataclasses
import html
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from aic2026.embedding.base import Embedder

logger = logging.getLogger(__name__)

DEFAULT_HEADROOM_MB: float = 3072.0  # ADR-0003 ~3 GB online headroom on the 5070


def frame_id_for(path: Path) -> str:
    """Video-unique frame id used to key/caption a keyframe: ``<videoDir>_<stem>``."""
    return f"{path.parent.name}_{path.stem}"


def sample_keyframes(kf_root: Path, n: int, *, seed: int = 0) -> list[Path]:
    """Deterministically sample ``n`` ``*.jpg`` paths under ``kf_root``."""
    import random

    imgs = sorted(kf_root.rglob("*.jpg"))
    if not imgs:
        raise ValueError(f"no .jpg under {kf_root}")
    if n >= len(imgs):
        return imgs
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(imgs)), n))
    return [imgs[i] for i in idx]


def encode_images(encoder: Embedder, paths: Sequence[Path], *, batch_size: int = 32) -> np.ndarray:
    """Batch ``encode_image`` into a single ``(n, dim)`` matrix."""
    if not paths:
        return np.zeros((0, encoder.dim), dtype=np.float32)
    out = np.zeros((len(paths), encoder.dim), dtype=np.float32)
    for start in range(0, len(paths), batch_size):
        batch = list(paths[start : start + batch_size])
        vecs = np.asarray(encoder.encode_image(batch), dtype=np.float32)
        out[start : start + len(batch)] = vecs
    return out


def topk_indices(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` doc indices per query by dot product (rows are L2-normalised)."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return np.zeros((query_vecs.shape[0], 0), dtype=int)
    sims = query_vecs @ doc_vecs.T  # (nq, nd)
    kk = min(k, doc_vecs.shape[0])
    part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    return part[rows, order]


# --- deployability ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeployStat:
    """Per-encoder deployability measurement (SPEC-0025 SS 4)."""

    model_id: str
    dim: int
    quant: str  # "fp16" | "int4" | "cpu"
    vram_mb: float | None  # peak VRAM of an encode_text forward (None on CPU)
    latency_p50_ms: float
    latency_p95_ms: float
    fits_5070_headroom: bool | None  # vram_mb <= headroom_mb; None when VRAM unknown

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def measure_deployability(
    encoder: Embedder,
    sample_texts: Sequence[str],
    *,
    headroom_mb: float = DEFAULT_HEADROOM_MB,
    repeats: int = 20,
    quant: str = "fp16",
) -> DeployStat:
    """Measure ``encode_text`` latency (p50/p95) + peak VRAM for ``encoder``.

    VRAM is read via ``torch.cuda.max_memory_allocated`` around a forward when
    torch+CUDA are available; otherwise ``None`` (CPU/CI path). ``quant`` is a
    label for which instance was passed (the caller constructs fp16 vs int4).
    """
    texts = list(sample_texts) or ["xin chao"]
    encoder.encode_text(texts)  # warmup (lazy weights, cudnn autotune)

    vram_mb, fits = _peak_vram_mb(encoder, texts), None
    if vram_mb is not None:
        fits = vram_mb <= headroom_mb

    samples: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        encoder.encode_text(texts)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[min(len(samples) - 1, round(0.95 * (len(samples) - 1)))]

    return DeployStat(
        model_id=encoder.model_id,
        dim=encoder.dim,
        quant=quant if vram_mb is not None else "cpu",
        vram_mb=vram_mb,
        latency_p50_ms=round(p50, 3),
        latency_p95_ms=round(p95, 3),
        fits_5070_headroom=fits,
    )


def _peak_vram_mb(encoder: Embedder, texts: list[str]) -> float | None:
    """Peak CUDA VRAM (MB) of one ``encode_text`` forward, or ``None`` on CPU."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats()
    encoder.encode_text(texts)
    return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)


# --- qualitative side-by-side -------------------------------------------------


def run_qualitative(
    encoders: Mapping[str, Embedder],
    query_texts: Sequence[str],
    doc_paths: Sequence[Path],
    *,
    top_k: int = 5,
    out_html: Path,
    batch_size: int = 32,
) -> dict[str, list[list[tuple[str, str, float]]]]:
    """Index ``doc_paths`` per encoder, take top-``k`` per query, write an HTML
    contact sheet, and return ``{encoder_label: [per-query [(frame_id, path,
    score), ...]]}`` for testability.
    """
    if not query_texts:
        raise ValueError("no query_texts")
    if not doc_paths:
        raise ValueError("no doc_paths")
    paths = [Path(p) for p in doc_paths]
    frame_ids = [frame_id_for(p) for p in paths]

    results: dict[str, list[list[tuple[str, str, float]]]] = {}
    for label, enc in encoders.items():
        doc_vecs = encode_images(enc, paths, batch_size=batch_size)
        qv = np.asarray(enc.encode_text(list(query_texts)), dtype=np.float32)
        sims = qv @ doc_vecs.T if doc_vecs.size else np.zeros((len(query_texts), 0))
        top = topk_indices(qv, doc_vecs, top_k)
        per_query: list[list[tuple[str, str, float]]] = []
        for qi in range(len(query_texts)):
            hits = [
                (frame_ids[int(di)], str(paths[int(di)]), float(sims[qi, int(di)]))
                for di in top[qi]
            ]
            per_query.append(hits)
        results[label] = per_query

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(_render_html(list(query_texts), results, top_k), encoding="utf-8")
    logger.info("wrote %s (%d queries x %d encoders)", out_html, len(query_texts), len(encoders))
    return results


def _render_html(
    query_texts: list[str],
    results: Mapping[str, list[list[tuple[str, str, float]]]],
    top_k: int,
) -> str:
    parts: list[str] = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Encoder bake-off (SPEC-0025)</title>",
        "<style>body{font-family:sans-serif;margin:16px}"
        ".q{margin:24px 0;border-top:2px solid #ccc;padding-top:8px}"
        ".row{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}"
        ".cell{font-size:11px;text-align:center}"
        "img{width:160px;height:90px;object-fit:cover;border:1px solid #999}"
        ".enc{font-weight:bold;margin-top:8px}</style>",
        f"<h1>Encoder bake-off - top-{top_k} per query</h1>",
        "<p>Directional screen, no ground truth (SPEC-0025). Eyeball which "
        "encoder surfaces the relevant frame.</p>",
    ]
    for qi, qtext in enumerate(query_texts):
        parts.append(f"<div class='q'><h3>Query {qi + 1}</h3><p>{html.escape(qtext)}</p>")
        for label, per_query in results.items():
            parts.append(f"<div class='enc'>{html.escape(label)}</div><div class='row'>")
            for frame_id, path, score in per_query[qi]:
                parts.append(
                    "<div class='cell'>"
                    f"<img src='{html.escape(path)}' alt='{html.escape(frame_id)}'><br>"
                    f"{html.escape(frame_id)}<br>{score:+.3f}</div>"
                )
            parts.append("</div>")
        parts.append("</div>")
    return "\n".join(parts)
