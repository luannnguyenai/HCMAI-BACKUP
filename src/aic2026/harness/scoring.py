# Implements SPEC-0001 SS 4 (scoring) and proposal 05 SS 11 (formulas).
# Implements SPEC-0020 SS 3-4 (NDCG@10).
"""Pure scoring functions.

All functions in this module are pure: no I/O, no hidden state, deterministic
given their inputs. They are unit-tested in `tests/unit/test_scoring.py`.

Score formulas trace to `docs/proposals/05-evaluation-harness.md` and to the
LSC review (see `docs/research-notes/02-lsc-vbs-systems-deep-dive.md` SS 2).
"""

from __future__ import annotations

import math

from aic2026.models.submission import Submission
from aic2026.models.task import GroundTruth, MockTask, TaskType

# KIS / QA scoring constants per proposal 05 SS 11 (formulas inherited
# from the LSC scoring rules).
_KIS_TIME_PENALTY: float = 50.0
_KIS_WRONG_SUBMISSION_PENALTY: float = 10.0
_AD_HOC_BASE: float = 100.0


# --- Helpers ----------------------------------------------------------------


def _normalise_text(text: str) -> str:
    """Cheap normalisation for QA answer matching.

    Lower-cases, strips, and collapses internal whitespace. We deliberately
    do **not** strip Vietnamese diacritics here (see ADR-0007 C1
    DiacriticBERT and `docs/research-notes/05-baseline-2025-analysis.md`
    SS 4.1 for why diacritic-stripping at the retrieval layer hurts).
    """
    return " ".join(text.strip().lower().split())


# --- Ranked-retrieval metrics ----------------------------------------------


def r_at_k(submissions: list[Submission], correct_ids: set[str], k: int) -> float:
    """Binary recall at k: 1.0 if at least one of `correct_ids` is in the
    top-k of `submissions`, else 0.0. Used for KIS-style targets.

    Submissions are assumed to be sorted by `rank` ascending.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if not correct_ids:
        return 0.0
    top_k = submissions[:k]
    return 1.0 if any(s.frame_id in correct_ids for s in top_k) else 0.0


def mrr(submissions: list[Submission], correct_ids: set[str]) -> float:
    """Reciprocal rank of the first correct submission, or 0 if none."""
    if not correct_ids:
        return 0.0
    for s in submissions:
        if s.frame_id is not None and s.frame_id in correct_ids:
            return 1.0 / s.rank
    return 0.0


def adhoc_recall_at_k(
    submissions: list[Submission],
    correct_ids: set[str],
    k: int,
) -> float:
    """Fraction of `correct_ids` covered by the top-k submissions.

    Differs from `r_at_k` (binary) by measuring how *many* relevant items we
    retrieved, not just whether at least one made the cut. This is the
    Ad-hoc-flavoured recall@k.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if not correct_ids:
        return 0.0
    top_k = submissions[:k]
    hits = sum(1 for s in top_k if s.frame_id in correct_ids)
    return hits / len(correct_ids)


# --- NDCG (SPEC-0020) ------------------------------------------------------


def dcg_at_k(submissions: list[Submission], correct_ids: set[str], k: int) -> float:
    """Discounted cumulative gain at k with binary gains (SPEC-0020 SS 3).

    DCG@k = sum_{i=1..k} rel_i / log2(i + 1), where rel_i is 1 if the i-th
    submission's frame_id is in `correct_ids`, else 0. Submissions are assumed
    sorted by `rank` ascending (rank 1 first). Items beyond k do not contribute.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    dcg = 0.0
    for position, s in enumerate(submissions[:k], start=1):
        if s.frame_id is not None and s.frame_id in correct_ids:
            dcg += 1.0 / math.log2(position + 1)
    return dcg


def ndcg_at_k(submissions: list[Submission], correct_ids: set[str], k: int) -> float:
    """Normalised DCG at k with binary gains (SPEC-0020 SS 3-4).

    NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the ideal ranking
    (all relevant items first):
        IDCG@k = sum_{i=1..min(|correct_ids|, k)} 1 / log2(i + 1).
    Returns 0.0 when there are no relevant items or IDCG@k is 0.
    Binary gains are used because the harness ground truth is binary; graded
    relevance is deferred (SPEC-0020 SS 9 Q1).
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if not correct_ids:
        return 0.0
    dcg = dcg_at_k(submissions, correct_ids, k)
    n_ideal = min(len(correct_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# --- Task scores (the LSC formulas) ----------------------------------------


def kis_score(time_to_correct_s: float | None, wrong_submissions: int, time_limit_s: int) -> float:
    """LSC KIS score.

    Formula (proposal 05 SS 11):
        score = max(0, 100 - 50 * (t / T) - 10 * w)
    where t = seconds elapsed at first correct submission, T = task time
    limit, w = number of incorrect submissions before the correct one.

    If `time_to_correct_s is None`, the task was not solved -> score 0.
    """
    if time_to_correct_s is None:
        return 0.0
    if time_limit_s <= 0:
        raise ValueError(f"time_limit_s must be positive; got {time_limit_s}")
    if wrong_submissions < 0:
        raise ValueError(f"wrong_submissions must be non-negative; got {wrong_submissions}")
    fraction = min(time_to_correct_s / time_limit_s, 1.0)
    raw = 100.0 - _KIS_TIME_PENALTY * fraction - _KIS_WRONG_SUBMISSION_PENALTY * wrong_submissions
    return max(0.0, raw)


def adhoc_score(correct: int, incorrect: int, total_relevant: int) -> float:
    """LSC Ad-hoc score.

    Formula (proposal 05 SS 11):
        score = 100 * correct / (correct + incorrect / 2) * correct / total

    `total_relevant` is the pool size aggregated across teams (we use the
    per-task ground-truth size as a stand-in in the local harness).
    """
    if correct < 0 or incorrect < 0 or total_relevant < 0:
        raise ValueError("correct, incorrect, total_relevant must all be non-negative")
    if correct == 0 or total_relevant == 0:
        return 0.0
    denom = correct + incorrect / 2.0
    return _AD_HOC_BASE * (correct / denom) * (correct / total_relevant)


# --- QA answer matching ----------------------------------------------------


def qa_correct(submission_text: str, ground_truth: GroundTruth) -> bool:
    """QA correctness: substring match against canonical + acceptable variants.

    Per SPEC-0001 SS 3.1 the canonical phrasing is at
    `ground_truth.qa_answer`; alternate phrasings live in
    `ground_truth.qa_answer_acceptable`. We lowercase, strip, and collapse
    whitespace on both sides and ask whether *any* acceptable answer is a
    substring of the candidate (or vice versa, for short candidates).
    """
    if not submission_text or not ground_truth.qa_answer:
        return False
    candidate = _normalise_text(submission_text)
    targets = [ground_truth.qa_answer, *ground_truth.qa_answer_acceptable]
    for target in targets:
        t = _normalise_text(target)
        if not t:
            continue
        if t in candidate or candidate in t:
            return True
    return False


# --- TRAKE correctness -----------------------------------------------------


def trake_correct(submissions: list[Submission], ground_truth: GroundTruth) -> bool:
    """TRAKE: the first 4 submissions must match the 4 ground-truth frames
    in the *exact* order.
    """
    if not ground_truth.trake_frame_ids or len(ground_truth.trake_frame_ids) != 4:
        return False
    if len(submissions) < 4:
        return False
    for idx, expected in enumerate(ground_truth.trake_frame_ids):
        if submissions[idx].frame_id != expected:
            return False
    return True


# --- One-stop task scorer used by the runner -------------------------------


def score_task(
    task: MockTask,
    submissions: list[Submission],
    elapsed_ms: float,
) -> dict[str, float | int | bool | None]:
    """Score one (task, submissions) pair.

    Returns a flat dict with the per-task metric values; the runner converts
    this into a `TaskMetrics` record. Splitting the conversion lets the
    scorer remain Pydantic-free.
    """
    gt = task.ground_truth
    elapsed_s = elapsed_ms / 1000.0

    correct_id_set: set[str] = set()
    match task.task_type:
        case TaskType.KIS:
            correct_id_set = set(gt.kis_frame_ids or [])
        case TaskType.AD_HOC:
            correct_id_set = set(gt.adhoc_frame_ids or [])
        case TaskType.TRAKE:
            correct_id_set = set(gt.trake_frame_ids or [])
        case TaskType.QA:
            correct_id_set = set()

    r1 = r_at_k(submissions, correct_id_set, 1) if correct_id_set else 0.0
    r5 = r_at_k(submissions, correct_id_set, 5) if correct_id_set else 0.0
    r10 = r_at_k(submissions, correct_id_set, 10) if correct_id_set else 0.0
    mean_rr = mrr(submissions, correct_id_set) if correct_id_set else 0.0
    ndcg10 = ndcg_at_k(submissions, correct_id_set, 10) if correct_id_set else 0.0

    out: dict[str, float | int | bool | None] = {
        "r_at_1": r1,
        "r_at_5": r5,
        "r_at_10": r10,
        "mrr": mean_rr,
        "ndcg_at_10": ndcg10,
        "end_to_end_ms": elapsed_ms,
        "wrong_submissions": 0,
        "time_to_first_correct_ms": None,
        "kis_score": None,
        "adhoc_score": None,
        "adhoc_correct": None,
        "adhoc_incorrect": None,
        "ok": False,
    }

    if task.task_type is TaskType.KIS:
        if r1 == 1.0:
            ttc_ms = elapsed_ms
            wrong = 0
        elif r10 == 1.0:
            ttc_ms = elapsed_ms
            wrong = next(
                (s.rank - 1 for s in submissions if s.frame_id in correct_id_set),
                0,
            )
        else:
            ttc_ms = None
            wrong = len(submissions)
        out["time_to_first_correct_ms"] = ttc_ms
        out["wrong_submissions"] = wrong
        out["kis_score"] = kis_score(
            time_to_correct_s=(ttc_ms / 1000.0) if ttc_ms is not None else None,
            wrong_submissions=wrong,
            time_limit_s=task.time_limit_seconds,
        )
        out["ok"] = ttc_ms is not None

    elif task.task_type is TaskType.AD_HOC:
        relevant = correct_id_set
        top10_ids = [s.frame_id for s in submissions[:10] if s.frame_id]
        correct = sum(1 for fid in top10_ids if fid in relevant)
        incorrect = sum(1 for fid in top10_ids if fid not in relevant)
        out["adhoc_correct"] = correct
        out["adhoc_incorrect"] = incorrect
        out["adhoc_score"] = adhoc_score(correct, incorrect, len(relevant) or 1)
        out["wrong_submissions"] = incorrect
        out["ok"] = correct > 0

    elif task.task_type is TaskType.TRAKE:
        ok = trake_correct(submissions, gt)
        out["ok"] = ok
        if ok:
            out["time_to_first_correct_ms"] = elapsed_ms
            out["kis_score"] = kis_score(
                time_to_correct_s=elapsed_ms / 1000.0,
                wrong_submissions=0,
                time_limit_s=task.time_limit_seconds,
            )
        else:
            out["kis_score"] = 0.0
            out["wrong_submissions"] = 1

    elif task.task_type is TaskType.QA:
        # Top-1 candidate's text is what we judge.
        top1_text = submissions[0].text if submissions and submissions[0].text else ""
        ok = qa_correct(top1_text, gt)
        out["ok"] = ok
        out["r_at_1"] = 1.0 if ok else 0.0
        out["r_at_5"] = out["r_at_1"]
        out["r_at_10"] = out["r_at_1"]
        out["mrr"] = out["r_at_1"]
        out["ndcg_at_10"] = out["r_at_1"]
        if ok:
            out["time_to_first_correct_ms"] = elapsed_ms
            out["kis_score"] = kis_score(
                time_to_correct_s=elapsed_ms / 1000.0,
                wrong_submissions=0,
                time_limit_s=task.time_limit_seconds,
            )
        else:
            out["kis_score"] = 0.0
            out["wrong_submissions"] = 1

    _ = elapsed_s  # currently unused; reserved for future per-stage timing
    return out
