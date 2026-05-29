# Unit tests for the pure scoring functions in aic2026.harness.scoring.
# NDCG tests prove SPEC-0020 AC1, AC3, AC4.

from __future__ import annotations

import math

import pytest

from aic2026.harness.scoring import (
    adhoc_recall_at_k,
    adhoc_score,
    kis_score,
    mrr,
    ndcg_at_k,
    qa_correct,
    r_at_k,
    score_task,
    trake_correct,
)
from aic2026.models.submission import Submission
from aic2026.models.task import GroundTruth, MockTask, TaskType


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


# --- NDCG@10 (SPEC-0020) ---------------------------------------------------


def _idcg(n_relevant: int) -> float:
    return sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))


def test_ndcg_perfect_ranking_is_one_AC1() -> None:
    # All relevant items first -> DCG == IDCG -> NDCG == 1.
    relevant = {"a", "b"}
    assert ndcg_at_k(_subs("a", "b", "c"), relevant, 10) == pytest.approx(1.0)


def test_ndcg_single_relevant_rank_discount_AC1() -> None:
    # Single relevant: NDCG == 1 / log2(rank + 1) since IDCG == 1.
    assert ndcg_at_k(_subs("a", "b"), {"a"}, 10) == pytest.approx(1.0)
    # Correct item at rank 3 -> 1 / log2(4) == 0.5.
    assert ndcg_at_k(_subs("x", "y", "a", "z"), {"a"}, 10) == pytest.approx(0.5)


def test_ndcg_partial_multi_relevant_AC1() -> None:
    # relevant {a,b,c,d}; retrieved a (r1) and c (r3) in top-10.
    relevant = {"a", "b", "c", "d"}
    subs = _subs("a", "x", "c", "y", "z")
    expected = (1.0 / math.log2(2) + 1.0 / math.log2(4)) / _idcg(4)
    assert ndcg_at_k(subs, relevant, 10) == pytest.approx(expected)


def test_ndcg_empty_relevant_is_zero_AC3() -> None:
    assert ndcg_at_k(_subs("a", "b"), set(), 10) == 0.0


def test_ndcg_ignores_items_beyond_k_AC3() -> None:
    # The only relevant item sits at rank 11; NDCG@10 must not see it.
    subs = _subs(*[f"f{i}" for i in range(10)], "a")  # "a" is at rank 11
    assert ndcg_at_k(subs, {"a"}, 10) == 0.0


def test_ndcg_invalid_k_raises_AC3() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(_subs("a"), {"a"}, 0)


def _qa_task() -> MockTask:
    return MockTask(
        task_id="QA-0001",
        task_type=TaskType.QA,
        query_vi="co bao nhieu chiec xe?",
        time_limit_seconds=180,
        ground_truth=GroundTruth(qa_answer="ba", qa_answer_acceptable=["3"]),
    )


def test_score_task_qa_ndcg_collapses_to_correctness_AC4() -> None:
    task = _qa_task()
    correct = score_task(task, [Submission(rank=1, score=1.0, text="ba chiec")], 1000.0)
    assert correct["ndcg_at_10"] == 1.0
    wrong = score_task(task, [Submission(rank=1, score=1.0, text="hai")], 1000.0)
    assert wrong["ndcg_at_10"] == 0.0
