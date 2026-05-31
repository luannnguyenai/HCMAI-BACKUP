# Implements SPEC-0014 section 3 + AC4 (the C1 degradation@10 sweep) + the
# head-as-encoder ship-gate comparison (SPEC-0014 follow-up: AC7-AC9).
"""C1 synthetic noise-sweep eval (proposal 05 part 13.1) + the ship-gate.

``degradation@k = R@k(noisy) / R@k(clean)``: for each clean query we build noisy
variants, retrieve the top-k against the index of clean targets, and measure how
much retrieval degrades. With the C1 head on, target is ``>= 0.85``; a BGE-M3-only
baseline is expected ``~0.65-0.75``.

This module exposes three layers:

  * ``degradation_at_k`` accepts a ``Retriever`` *or* an ``Embedder`` (back-compat
    with the AC4 DummyEmbedder path). It picks ranks off a single ``(nq, nd)``
    score matrix, so it works identically for cosine and MaxSim.
  * ``build_heldout_queries`` harvests the same public sources used by the
    contrastive corpus and removes anything seen at training time -- a disjoint
    held-out set keyed off the training Parquet's ``anchor_clean`` column.
  * ``compare_c1_vs_baselines`` is the ship-gate: it builds three retrievers
    (C1 on; raw BGE-M3 MaxSim; BGE-M3 dense mean-pool) sharing one backbone,
    runs the same noise sweep on each, and returns the three numbers plus a
    pass/fail verdict against the SPEC-0014 / proposal 05 SS 13.1 target.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aic2026.embedding.base import Embedder
from aic2026.eval.retrievers import DenseRetriever, Retriever
from aic2026.train.diacritic_noise import NoiseMode, noise

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from aic2026.train.diacritic_bert import DiacriticHead, TokenEncoder

logger = logging.getLogger(__name__)

DEFAULT_MODES: tuple[NoiseMode, ...] = (
    NoiseMode.DROP_ALL,
    NoiseMode.RANDOM_DROP,
    NoiseMode.TONE_SWAP,
    NoiseMode.MIXED,
)

SHIP_GATE_TARGET: float = 0.85
"""Proposal 05 SS 13.1: with C1 on, target ``degradation@10 >= 0.85``."""


def _topk_indices(row: np.ndarray, k: int) -> set[int]:
    """Return the indices of the top-``k`` scores in ``row`` (descending)."""
    if k >= row.size:
        return set(range(row.size))
    part = np.argpartition(-row, k - 1)[:k]
    return set(part.tolist())


def _as_retriever(obj: Retriever | Embedder) -> Retriever:
    """Accept either a ``Retriever`` (preferred) or an ``Embedder`` (back-compat).

    AC4 keeps passing ``DummyEmbedder`` directly; new callers pass a
    ``Retriever``. We dispatch by attribute: ``score`` -> already a retriever;
    otherwise wrap the embedder in ``DenseRetriever``.
    """
    if hasattr(obj, "score"):
        return obj  # type: ignore[return-value]
    if hasattr(obj, "encode_text"):
        return DenseRetriever(obj)  # type: ignore[arg-type]
    raise TypeError(f"expected Retriever or Embedder, got {type(obj).__name__}")


def _build_noisy(queries: Sequence[str], mode: NoiseMode, seed: int) -> list[str]:
    return [noise(q, mode, rng=random.Random(f"{seed}\x00{mode.value}\x00{q}")) for q in queries]


def _hits_at_k(scores: np.ndarray, k: int) -> int:
    """Count rows whose own column index is among the row's top-k."""
    n = scores.shape[0]
    hits = 0
    for i in range(n):
        if i in _topk_indices(scores[i], k):
            hits += 1
    return hits


def degradation_at_k(
    clean_queries: Sequence[str],
    encoder: Retriever | Embedder,
    *,
    k: int = 10,
    modes: Sequence[NoiseMode] = DEFAULT_MODES,
    seed: int = 0,
) -> dict[str, float]:
    """Return ``degradation@k`` per mode and ``"overall"`` (all in [0, 1]).

    Deterministic given ``(clean_queries, encoder, seed)``. The index is the set
    of clean queries themselves; each query's own row is the retrieval target.

    ``encoder`` may be a ``Retriever`` (new ship-gate path) or an ``Embedder``
    (back-compat; AC4). For an ``Embedder`` we wrap in ``DenseRetriever``, which
    matches the historic ``q @ d.T`` semantics exactly.
    """
    queries = list(clean_queries)
    n = len(queries)
    if n == 0:
        raise ValueError("clean_queries is empty")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")

    retriever = _as_retriever(encoder)

    clean_scores = np.asarray(retriever.score(queries, queries), dtype=np.float32)
    if clean_scores.shape != (n, n):
        raise ValueError(
            f"retriever returned wrong shape {clean_scores.shape}; expected ({n}, {n})"
        )
    rk_clean = _hits_at_k(clean_scores, k) / n

    out: dict[str, float] = {}
    for mode in modes:
        noisy = _build_noisy(queries, mode, seed)
        noisy_scores = np.asarray(retriever.score(noisy, queries), dtype=np.float32)
        if noisy_scores.shape != (n, n):
            raise ValueError(
                f"retriever returned wrong shape {noisy_scores.shape}; expected ({n}, {n})"
            )
        rk_noisy = _hits_at_k(noisy_scores, k) / n
        out[mode.value] = (rk_noisy / rk_clean) if rk_clean > 0 else 0.0
    out["overall"] = sum(out[m.value] for m in modes) / len(modes)
    return out


# --- held-out query harvester -------------------------------------------------


def _read_training_anchors(pairs_path: Path) -> set[str]:
    """Return the case/space-normalised ``anchor_clean`` set of a training Parquet."""
    from aic2026.train.diacritic_corpus import _normalize, read_pairs

    rows = read_pairs(pairs_path)
    return {_normalize(str(r["anchor_clean"])).casefold() for r in rows}


def build_heldout_queries(
    n: int,
    *,
    exclude_corpus: Path | None = None,
    seed: int = 0,
    max_per_source: int | None = None,
) -> list[str]:
    """Harvest ``n`` clean Vietnamese strings disjoint from the training corpus.

    Uses the same public sources as the contrastive corpus (Wikipedia + KTVIC),
    deduped and excluded against ``anchor_clean`` from ``exclude_corpus``. The
    held-out seed is mixed in so a different sample is selected from the
    upstream sources than the training run picked.
    """
    from aic2026.train.diacritic_corpus import (
        DEFAULT_SOURCES,
        _collect_clean,
        _dedup,
        _normalize,
    )

    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")

    used: list[str] = []
    skipped: list[str] = []
    raw = _collect_clean(DEFAULT_SOURCES, max_per_source=max_per_source, used=used, skipped=skipped)
    clean = _dedup(raw)
    excluded: set[str] = _read_training_anchors(exclude_corpus) if exclude_corpus else set()

    held: list[str] = []
    seen: set[str] = set()
    for s in clean:
        key = _normalize(s).casefold()
        if key in excluded or key in seen:
            continue
        seen.add(key)
        held.append(s)

    if not held:
        raise ValueError(
            f"held-out harvest produced 0 disjoint strings (used={used} skipped={skipped})"
        )

    rng = random.Random(f"{seed}\x00heldout")
    rng.shuffle(held)
    return held[:n]


# --- the three-way ship-gate comparison --------------------------------------


def _verdict(
    c1_overall: float,
    base_maxsim_overall: float,
    base_dense_overall: float,
    *,
    target: float = SHIP_GATE_TARGET,
) -> dict[str, Any]:
    """Build the ship-gate verdict block."""
    passes_abs = c1_overall >= target
    beats_max = c1_overall > base_maxsim_overall
    beats_dense = c1_overall > base_dense_overall
    return {
        "target": target,
        "c1_overall": c1_overall,
        "baseline_maxsim_overall": base_maxsim_overall,
        "baseline_dense_overall": base_dense_overall,
        "passes_absolute": passes_abs,
        "beats_baseline_maxsim": beats_max,
        "beats_baseline_dense": beats_dense,
        "passes_ship_gate": passes_abs and beats_max and beats_dense,
    }


def compare_c1_vs_baselines(
    clean_queries: Sequence[str],
    *,
    backbone: TokenEncoder,
    head: DiacriticHead,
    k: int = 10,
    modes: Sequence[NoiseMode] = DEFAULT_MODES,
    seed: int = 0,
    target: float = SHIP_GATE_TARGET,
) -> dict[str, Any]:
    """Run degradation@k for C1-on vs raw-MaxSim vs dense baselines.

    ``backbone`` and ``head`` must already be loaded (use
    ``aic2026.eval.retrievers.load_head``; for BGE-M3 use ``BgeM3Backbone``).
    Returns a dict with the three per-retriever score dicts plus a
    ``ship_gate`` verdict block.
    """
    from aic2026.eval.retrievers import BgeM3DenseEmbedder, MaxSimRetriever

    c1 = MaxSimRetriever(backbone, head=head)
    base_maxsim = MaxSimRetriever(backbone, head=None)
    base_dense = DenseRetriever(BgeM3DenseEmbedder(backbone))

    c1_scores = degradation_at_k(clean_queries, c1, k=k, modes=modes, seed=seed)
    bm_scores = degradation_at_k(clean_queries, base_maxsim, k=k, modes=modes, seed=seed)
    bd_scores = degradation_at_k(clean_queries, base_dense, k=k, modes=modes, seed=seed)

    return {
        "k": k,
        "n_queries": len(clean_queries),
        "modes": [m.value for m in modes],
        "c1_on": c1_scores,
        "baseline_maxsim": bm_scores,
        "baseline_dense": bd_scores,
        "ship_gate": _verdict(
            c1_scores["overall"],
            bm_scores["overall"],
            bd_scores["overall"],
            target=target,
        ),
    }
