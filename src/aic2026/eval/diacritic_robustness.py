# Implements SPEC-0014 section 3 + AC4 (the C1 degradation@10 sweep).
"""C1 synthetic noise-sweep eval (proposal 05 part 13.1).

``degradation@k = R@k(noisy) / R@k(clean)``: for each clean query we build noisy
variants, retrieve the top-k against the index of clean targets, and measure how
much retrieval degrades. With the C1 head on, target is ``>= 0.85``; a BGE-M3-only
baseline is expected ``~0.65-0.75``.

The encoder is **pluggable** (any ``Embedder``): ``DummyEmbedder`` for CI, the
trained C1 head on the box. Pure numpy - no torch dependency here.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from aic2026.embedding.base import Embedder
from aic2026.train.diacritic_noise import NoiseMode, noise

DEFAULT_MODES: tuple[NoiseMode, ...] = (
    NoiseMode.DROP_ALL,
    NoiseMode.RANDOM_DROP,
    NoiseMode.TONE_SWAP,
    NoiseMode.MIXED,
)


def _topk_hit(query_vec: np.ndarray, index: np.ndarray, target: int, k: int) -> bool:
    """True if ``target`` is among the top-``k`` index rows by cosine similarity."""
    sims = index @ query_vec
    if k >= len(sims):
        return True
    topk = np.argpartition(-sims, k - 1)[:k]
    return target in set(topk.tolist())


def degradation_at_k(
    clean_queries: Sequence[str],
    encoder: Embedder,
    *,
    k: int = 10,
    modes: Sequence[NoiseMode] = DEFAULT_MODES,
    seed: int = 0,
) -> dict[str, float]:
    """Return ``degradation@k`` per mode and ``"overall"`` (all in [0, 1]).

    Deterministic given ``(clean_queries, encoder, seed)``. The index is the set
    of clean queries themselves; each query's own row is the retrieval target.
    """
    queries = list(clean_queries)
    n = len(queries)
    if n == 0:
        raise ValueError("clean_queries is empty")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")

    index = np.asarray(encoder.encode_text(queries), dtype=np.float32)

    rk_clean = sum(_topk_hit(index[i], index, i, k) for i in range(n)) / n
    out: dict[str, float] = {}
    for mode in modes:
        hits = 0
        for i, q in enumerate(queries):
            noisy = noise(q, mode, rng=random.Random(f"{seed}\x00{mode.value}\x00{q}"))
            nv = np.asarray(encoder.encode_text([noisy])[0], dtype=np.float32)
            hits += _topk_hit(nv, index, i, k)
        rk_noisy = hits / n
        out[mode.value] = (rk_noisy / rk_clean) if rk_clean > 0 else 0.0
    out["overall"] = sum(out[m.value] for m in modes) / len(modes)
    return out
