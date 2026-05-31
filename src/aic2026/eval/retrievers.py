# Implements SPEC-0014 SS 3 (head-as-encoder retrieval for degradation@10).
"""Retriever abstractions for the C1 ship-gate eval.

The C1 head is a **late-interaction** scorer (ColBERT-style MaxSim over per-token
vectors), not a single-vector encoder. The existing ``degradation_at_k`` runs on
single-vector cosine, so C1 can't be evaluated against it directly. This module
adds a thin ``Retriever`` Protocol with two implementations:

  * ``DenseRetriever``: any ``Embedder`` -> ``encode_text`` both sides -> q @ d.T.
    Pure numpy; backwards-compatible with ``DummyEmbedder``.
  * ``MaxSimRetriever``: a frozen token encoder (e.g. BGE-M3) + an optional
    ``DiacriticHead``. Encodes queries and docs, applies the head if present,
    and reuses ``maxsim_scores`` to produce the score matrix - chunked over
    queries so the ``(nq, nd, Tq, Td)`` tensor stays bounded on real corpora.

Both implementations have lazy torch imports - this module imports cleanly on
CPU/CI without torch, and the heavy paths only activate when a torch-backed
encoder is constructed by the caller. ``BgeM3DenseEmbedder`` and ``load_head``
require torch + the ``train`` extra and are only used by the ship-gate path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from aic2026.embedding.base import Embedder, l2_normalize

if TYPE_CHECKING:  # pragma: no cover - typing-only
    import torch

    from aic2026.train.diacritic_bert import DiacriticHead, TokenEncoder

logger = logging.getLogger(__name__)


@runtime_checkable
class Retriever(Protocol):
    """A retriever returns an ``(nq, nd)`` similarity matrix for two text lists.

    Score magnitudes are not normalised across retrievers; only ranks matter for
    the degradation eval. Higher is better.
    """

    def score(self, queries: list[str], docs: list[str]) -> np.ndarray: ...


class DenseRetriever:
    """Wraps any single-vector ``Embedder``: ``score = q @ d.T`` after L2-norm.

    Pure numpy. The vectors from `Embedder.encode_text` are already L2-normalised
    by contract (SPEC-0004 SS 3), so the dot product is cosine similarity. We
    L2-normalise defensively in case a future encoder forgets.
    """

    def __init__(self, encoder: Embedder) -> None:
        self.encoder = encoder

    def score(self, queries: list[str], docs: list[str]) -> np.ndarray:
        q = l2_normalize(np.asarray(self.encoder.encode_text(queries), dtype=np.float32))
        d = l2_normalize(np.asarray(self.encoder.encode_text(docs), dtype=np.float32))
        return (q @ d.T).astype(np.float32, copy=False)


class MaxSimRetriever:
    """Late-interaction retriever: frozen token encoder + optional ``DiacriticHead``.

    Encodes queries and docs to ``(B, T, H)`` token embeddings, applies the
    L2-normalising head (when supplied; otherwise the raw last-hidden tokens
    are L2-normalised in-place), then computes a ``(nq, nd)`` MaxSim score
    matrix via the same primitive used at training time.

    Chunked over queries so the intermediate ``(chunk, nd, Tq, Td)`` tensor stays
    bounded; defaults to ``q_chunk=32``, fine for the held-out sets (a few
    hundred queries) we run on a lease.
    """

    def __init__(
        self,
        backbone: TokenEncoder,
        head: DiacriticHead | None = None,
        *,
        q_chunk: int = 32,
        device: str | None = None,
    ) -> None:
        import torch  # noqa: F401  (lazy guard; raises clearly if extra missing)
        from torch import nn as _nn  # noqa: F401

        self.backbone = backbone
        self.head = head
        self.q_chunk = max(1, int(q_chunk))
        self.device = device

    def _project(self, tok: torch.Tensor) -> torch.Tensor:
        """Apply the head if present; else L2-normalise the raw tokens."""
        import torch
        import torch.nn.functional as F

        if self.head is not None:
            return self.head(tok)
        return F.normalize(tok, dim=-1).to(dtype=torch.float32)

    def score(self, queries: list[str], docs: list[str]) -> np.ndarray:
        import torch

        from aic2026.train.diacritic_bert import maxsim_scores

        if not queries or not docs:
            return np.zeros((len(queries), len(docs)), dtype=np.float32)

        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        with torch.no_grad():
            de, dm = self.backbone.encode(docs)
            de, dm = de.to(dev), dm.to(dev)
            d_proj = self._project(de)

            rows: list[torch.Tensor] = []
            for start in range(0, len(queries), self.q_chunk):
                chunk = queries[start : start + self.q_chunk]
                qe, qm = self.backbone.encode(chunk)
                qe, qm = qe.to(dev), qm.to(dev)
                q_proj = self._project(qe)
                rows.append(maxsim_scores(q_proj, qm, d_proj, dm).cpu())

        return torch.cat(rows, dim=0).numpy().astype(np.float32, copy=False)


class BgeM3DenseEmbedder:
    """Mean-pooled, L2-normalised single-vector encoder over a ``BgeM3Backbone``.

    Implements ``Embedder.encode_text``: the live-system style baseline (one
    vector per query). ``encode_image`` raises - this is a text-only encoder
    surfaced for the C1 eval, not part of the offline image pipeline (ADR-0003).
    """

    def __init__(self, backbone: TokenEncoder, *, device: str | None = None) -> None:
        import torch  # noqa: F401

        self.backbone = backbone
        self.dim = int(backbone.hidden_size)
        self.model_id = "bge-m3-dense-meanpool"
        self.device = device

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            e, m = self.backbone.encode(texts)
            e, m = e.to(dev), m.to(dev)
            mask = m.unsqueeze(-1).to(dtype=e.dtype)  # (B, T, 1)
            summed = (e * mask).sum(dim=1)  # (B, H)
            denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            pooled = F.normalize(summed / denom, dim=-1)
        return pooled.cpu().numpy().astype(np.float32, copy=False)

    def encode_image(self, paths: list[Path]) -> np.ndarray:
        _ = paths  # text-only encoder; method present so the Embedder protocol is satisfied
        raise NotImplementedError("BgeM3DenseEmbedder is text-only (SPEC-0014 eval).")


def load_head(checkpoint: Path) -> DiacriticHead:
    """Rebuild a ``DiacriticHead`` from a saved ``head.pt`` and return it in eval mode.

    The checkpoint format is the one ``train_diacritic_head`` writes:
    ``{"head_state": ..., "in_dim": int, "config": {... "proj_dims": (h0, h1) ...}}``.
    """
    import torch

    from aic2026.train.diacritic_bert import DiacriticHead

    ckpt = torch.load(Path(checkpoint), weights_only=False, map_location="cpu")
    in_dim = int(ckpt["in_dim"])
    cfg = ckpt.get("config") or {}
    proj_dims = cfg.get("proj_dims") or (384, 384)
    h0, h1 = int(proj_dims[0]), int(proj_dims[1])
    head = DiacriticHead(in_dim, (h0, h1))
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head
