# Proves SPEC-0006 AC5: hits_to_submissions maps a ranked Hit list to
# list[Submission] (SPEC-0001) carrying rank/score and the global pk as the
# Submission.frame_id (the answer identity scored against ground truth), with
# contiguous 1-based ranks. Pure adapter; no Milvus needed.

from __future__ import annotations

from aic2026.index.milvus_store import hits_to_submissions
from aic2026.index.models import Hit
from aic2026.models.submission import Submission


def test_hits_map_to_submissions_AC5() -> None:
    hits = [
        Hit(pk="L25_V011_0003", frame_id="0003", video_id="L25_V011", score=0.91, rank=1),
        Hit(pk="L25_V011_0007", frame_id="0007", video_id="L25_V011", score=0.74, rank=2),
        Hit(pk="L07_V003_0001", frame_id="0001", video_id="L07_V003", score=0.51, rank=3),
    ]
    subs = hits_to_submissions(hits)

    assert all(isinstance(s, Submission) for s in subs)
    assert [s.rank for s in subs] == [1, 2, 3]  # contiguous 1-based
    # Submission.frame_id carries the global pk (matches global ground truth).
    assert [s.frame_id for s in subs] == [h.pk for h in hits]
    assert [s.score for s in subs] == [h.score for h in hits]
    assert all(s.text is None for s in subs)


def test_empty_hits_map_to_empty_AC5() -> None:
    assert hits_to_submissions([]) == []


def test_ranks_are_reindexed_contiguous_AC5() -> None:
    # A non-contiguous slice (ranks 5,6) is re-emitted as contiguous 1,2.
    hits = [
        Hit(pk="L01_V001_0004", frame_id="0004", video_id="L01_V001", score=0.4, rank=5),
        Hit(pk="L01_V001_0005", frame_id="0005", video_id="L01_V001", score=0.3, rank=6),
    ]
    subs = hits_to_submissions(hits)
    assert [s.rank for s in subs] == [1, 2]
    assert [s.score for s in subs] == [0.4, 0.3]
