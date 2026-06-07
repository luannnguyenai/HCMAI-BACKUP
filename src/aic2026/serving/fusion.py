# Implements SPEC-0026 SS 4 (single-lane passthrough + RRF combine, ADR-0008).
"""Combining one or more per-lane ranked lists into one ranked list.

Two modes (SPEC-0026 SS 4):

- `single_lane`  one lane, scores passed through (the locked MVP default).
- `rrf_fuse`     reciprocal-rank fusion over >= 2 lanes (ADR-0008 runtime
                 fallback): score(d) = sum_lanes 1 / (rrf_k + rank_lane(d)),
                 with the raw per-lane scores retained for the UI.

This is the MVP fusion surface only; the C2 learned fusion (SPEC-0015) is
ground-truth-blocked and out of scope. Pure data in / data out (operates on
SPEC-0006 `Hit` objects), so it is unit-testable with no Milvus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from aic2026.index.models import Hit


@dataclass(frozen=True)
class FusedItem:
    """One post-fusion result, carrying the raw per-lane scores."""

    pk: str
    video_id: str
    frame_id: str
    score: float  # fused score (IP for single, RRF score for rrf)
    per_lane: dict[str, float] = field(default_factory=dict)


def single_lane(lane: str, hits: Sequence[Hit]) -> list[FusedItem]:
    """Pass one lane's ranked hits through, recording the lane score."""
    return [
        FusedItem(
            pk=h.pk,
            video_id=h.video_id,
            frame_id=h.frame_id,
            score=h.score,
            per_lane={lane: h.score},
        )
        for h in hits
    ]


def rrf_fuse(
    per_lane_hits: Mapping[str, Sequence[Hit]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[FusedItem]:
    """Reciprocal-rank fusion over >= 2 lanes (ADR-0008).

    `per_lane_hits` maps lane name -> that lane's ranked hits (rank order =
    list order). Each document's fused score is the sum over the lanes it
    appears in of `1 / (rrf_k + rank)` (rank 1-based). The raw per-lane IP
    scores are retained. Ties break deterministically by `pk` so the order is
    stable across runs and reproducible in tests.
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive; got {rrf_k}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")

    fused_score: dict[str, float] = {}
    per_lane: dict[str, dict[str, float]] = {}
    rep: dict[str, Hit] = {}

    for lane, hits in per_lane_hits.items():
        for rank, h in enumerate(hits, start=1):
            fused_score[h.pk] = fused_score.get(h.pk, 0.0) + 1.0 / (rrf_k + rank)
            per_lane.setdefault(h.pk, {})[lane] = h.score
            rep.setdefault(h.pk, h)

    ordered = sorted(fused_score.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[FusedItem] = []
    for pk, score in ordered[:top_k]:
        h = rep[pk]
        out.append(
            FusedItem(
                pk=pk,
                video_id=h.video_id,
                frame_id=h.frame_id,
                score=score,
                per_lane=per_lane[pk],
            )
        )
    return out
