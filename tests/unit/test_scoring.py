# Unit tests for the pure scoring functions in aic2026.harness.scoring.

from __future__ import annotations

from aic2026.harness.scoring import (
    adhoc_recall_at_k,
    adhoc_score,
    kis_score,
    mrr,
    qa_correct,
    r_at_k,
    trake_correct,
)
from aic2026.models.submission import Submission
from aic2026.models.task import GroundTruth


def _subs(*frame_ids: str) -> list[Submission]:
    return [
        Submission(rank=i + 1, score=1.0 - i * 0.05, frame_id=f) for i, f in enumerate(frame_ids)
    ]


# --- r_at_k ----------------------------------------------------------------


def test_r_at_k_hit_in_top_1() -> None:
    assert r_at_k(_subs("a", "b", "c"), {"a"}, 1) == 1.0


def test_r_at_k_miss_in_top_1_but_hit_in_top_3() -> None:
    subs = _subs("x", "y", "a", "z")
    assert r_at_k(subs, {"a"}, 1) == 0.0
    assert r_at_k(subs, {"a"}, 3) == 1.0


def test_r_at_k_no_relevant_returns_zero() -> None:
    assert r_at_k(_subs("a", "b"), set(), 5) == 0.0


# --- mrr -------------------------------------------------------------------


def test_mrr_first_correct_at_rank_3() -> None:
    subs = _subs("x", "y", "a", "z")
    assert mrr(subs, {"a"}) == 1.0 / 3.0


def test_mrr_no_correct_returns_zero() -> None:
    assert mrr(_subs("x", "y", "z"), {"a"}) == 0.0


# --- adhoc_recall_at_k -----------------------------------------------------


def test_adhoc_recall_partial_coverage() -> None:
    relevant = {"a", "b", "c", "d"}
    subs = _subs("a", "x", "c", "y", "z")  # 2/4 in top-5
    assert adhoc_recall_at_k(subs, relevant, 5) == 0.5


# --- LSC KIS score ---------------------------------------------------------


def test_kis_score_immediate_no_wrong() -> None:
    # t = 0, w = 0  -> raw = 100
    assert kis_score(time_to_correct_s=0.0, wrong_submissions=0, time_limit_s=300) == 100.0


def test_kis_score_at_time_limit_half_credit() -> None:
    # t = T, w = 0  -> raw = 100 - 50 = 50
    assert kis_score(time_to_correct_s=300.0, wrong_submissions=0, time_limit_s=300) == 50.0


def test_kis_score_floor_at_zero() -> None:
    assert kis_score(time_to_correct_s=300.0, wrong_submissions=10, time_limit_s=300) == 0.0


def test_kis_score_none_means_unsolved_zero() -> None:
    assert kis_score(time_to_correct_s=None, wrong_submissions=99, time_limit_s=300) == 0.0


# --- LSC Ad-hoc score ------------------------------------------------------


def test_adhoc_score_perfect_precision_full_recall() -> None:
    # correct=10, incorrect=0, total=10  ->  100 * 10/10 * 10/10 = 100
    assert adhoc_score(correct=10, incorrect=0, total_relevant=10) == 100.0


def test_adhoc_score_half_recall_full_precision() -> None:
    # correct=5, incorrect=0, total=10  ->  100 * 5/5 * 5/10 = 50
    assert adhoc_score(correct=5, incorrect=0, total_relevant=10) == 50.0


def test_adhoc_score_some_wrong_submissions_dampen() -> None:
    # correct=5, incorrect=4, total=10
    # -> 100 * 5/(5 + 4/2) * 5/10 = 100 * 5/7 * 0.5 ~= 35.71
    got = adhoc_score(correct=5, incorrect=4, total_relevant=10)
    assert abs(got - (100.0 * 5.0 / 7.0 * 0.5)) < 1e-9


def test_adhoc_score_zero_correct_is_zero() -> None:
    assert adhoc_score(correct=0, incorrect=5, total_relevant=10) == 0.0


# --- QA correctness --------------------------------------------------------


def _qa_gt(answer: str, *acceptable: str) -> GroundTruth:
    return GroundTruth(qa_answer=answer, qa_answer_acceptable=list(acceptable))


def test_qa_substring_match() -> None:
    assert qa_correct("ba chiec", _qa_gt("ba", "3", "ba chiec")) is True


def test_qa_acceptable_variant_matches() -> None:
    assert qa_correct("3", _qa_gt("ba", "3")) is True


def test_qa_unrelated_answer_is_wrong() -> None:
    assert qa_correct("hai", _qa_gt("ba", "3")) is False


def test_qa_empty_inputs_are_safe() -> None:
    assert qa_correct("", _qa_gt("ba")) is False


# --- TRAKE correctness -----------------------------------------------------


def test_trake_correct_exact_order() -> None:
    gt = GroundTruth(trake_frame_ids=["a", "b", "c", "d"])
    assert trake_correct(_subs("a", "b", "c", "d", "x"), gt) is True


def test_trake_wrong_order_is_incorrect() -> None:
    gt = GroundTruth(trake_frame_ids=["a", "b", "c", "d"])
    assert trake_correct(_subs("a", "c", "b", "d"), gt) is False


def test_trake_short_list_is_incorrect() -> None:
    gt = GroundTruth(trake_frame_ids=["a", "b", "c", "d"])
    assert trake_correct(_subs("a", "b", "c"), gt) is False
