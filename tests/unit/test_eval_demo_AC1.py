# Proves SPEC-0014 SS 6 (live demo) AC1: CannedExample noise is deterministic,
# the canned runner ranks-then-tallies correctly given controlled retriever
# score matrices, formats Vietnamese blocks with the [TRÚNG] marker at the
# correct rank, and the interactive REPL is EOF-safe + counts queries it ran.
#
# No torch required - we drive the runners with hand-built FakeRetriever
# instances so the tests pin down the demo's *contract* (output structure +
# tally semantics) independently of any backbone behaviour.

from __future__ import annotations

import io

import numpy as np
import pytest

from aic2026.eval.demo import (
    CANNED_EXAMPLES,
    CannedExample,
    _build_index,
    _classify_verdict,
    _rank_of,
    _recommend_seed,
    format_example_block,
    run_canned,
    run_interactive,
    tune_seeds,
)
from aic2026.train.diacritic_noise import NoiseMode


class FakeRetriever:
    """A retriever that returns a pre-built (nq, nd) score matrix.

    The matrix is keyed by ``(queries_tuple, docs_tuple)`` and looked up on
    each call to ``score``; if not provided we return a zero matrix.
    """

    def __init__(self, score_fn) -> None:
        self._fn = score_fn

    def score(self, queries: list[str], docs: list[str]) -> np.ndarray:
        return np.asarray(self._fn(queries, docs), dtype=np.float32)


# ---- determinism + identity behaviour --------------------------------------


def test_canned_make_noised_is_deterministic_AC1() -> None:
    ex = CANNED_EXAMPLES[0]
    a = ex.make_noised()
    b = ex.make_noised()
    assert a == b
    # The noised form should differ from the clean target (drop_all on a
    # diacritic-heavy place name is a guaranteed change).
    assert a != ex.clean_target


def test_canned_clean_sanity_returns_clean_AC1() -> None:
    clean_ex = next(e for e in CANNED_EXAMPLES if e.mode is None)
    assert clean_ex.make_noised() == clean_ex.clean_target


def test_canned_examples_cover_v3_modes_AC1() -> None:
    """The canned set should exhibit at least DROP_ALL, CHAR_CONFUSE, MIXED_OCR,
    WORD_MERGE, HOMOPHONE_SWAP, and one clean-sanity example."""
    modes = {ex.mode for ex in CANNED_EXAMPLES}
    assert NoiseMode.DROP_ALL in modes
    assert NoiseMode.CHAR_CONFUSE in modes
    assert NoiseMode.MIXED_OCR in modes
    assert NoiseMode.WORD_MERGE in modes
    assert NoiseMode.HOMOPHONE_SWAP in modes
    assert None in modes


def test_canned_set_size_and_mixed_ocr_emphasis_AC1() -> None:
    """At least 10 examples, with mixed_ocr (the mode C1 wins on) most-represented.

    Locks the demo-scope decision: a 12-example set weighted toward mixed_ocr so
    the showcase spends most of its airtime on the realistic-worst-case mode.
    Also guards against duplicate ids (which would collide the per-example noise
    RNG) and a non-control final example (the clean-sanity control must stay last
    so it reads as the closing parity check)."""
    assert len(CANNED_EXAMPLES) >= 10
    ids = [e.id for e in CANNED_EXAMPLES]
    assert len(ids) == len(set(ids)), "canned example ids must be unique"

    n_mixed = sum(1 for e in CANNED_EXAMPLES if e.mode is NoiseMode.MIXED_OCR)
    assert n_mixed >= 3, "mixed_ocr should be the emphasised mode"

    # Exactly one clean-sanity control, and it is the last example.
    n_clean = sum(1 for e in CANNED_EXAMPLES if e.mode is None)
    assert n_clean == 1
    assert CANNED_EXAMPLES[-1].mode is None


def test_canned_every_noised_example_actually_changes_AC1() -> None:
    """Every non-control example's baked-in seed must produce visible noise.

    Guards the demo against a 'lucky' seed that leaves the query (nearly)
    unchanged - a non-event on stage. We require the noised form to differ from
    the clean target for all modes except the clean-sanity control."""
    for ex in CANNED_EXAMPLES:
        if ex.mode is None:
            assert ex.make_noised() == ex.clean_target
        else:
            assert ex.make_noised() != ex.clean_target, f"{ex.id} produced no noise"


# ---- index building ---------------------------------------------------------


def test_build_index_dedupes_and_appends_canned_AC1() -> None:
    pool = ["a", "b", "a", "c", ""]
    canned = [
        CannedExample(id="e1", vi_explanation="x", clean_target="d", mode=None),
        CannedExample(id="e2", vi_explanation="x", clean_target="a", mode=None),  # already in pool
    ]
    docs = _build_index(pool, canned)
    assert docs == ["a", "b", "c", "d"]


# ---- rank helper ------------------------------------------------------------


def test_rank_of_descending_AC1() -> None:
    # scores =      [0.1, 0.9, 0.3, 0.5]
    # rank order =  [1,   0,   3,   2]  (highest first)
    # so index 1 is #1, index 3 is #2, index 2 is #3, index 0 is #4
    row = np.array([0.1, 0.9, 0.3, 0.5], dtype=np.float32)
    assert _rank_of(row, 1) == 1
    assert _rank_of(row, 3) == 2
    assert _rank_of(row, 2) == 3
    assert _rank_of(row, 0) == 4


# ---- format_example_block ---------------------------------------------------


def test_format_example_block_marks_hit_AC1() -> None:
    docs = ["aaa", "bbb", "ccc"]
    # one retriever, target "bbb" -> highest score is "bbb"
    row = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    out = format_example_block(
        index_1based=1,
        n_total=3,
        explanation="Giải thích",
        clean="bbb",
        noised="bbb-noised",
        retriever_results={"R1": row},
        docs=docs,
        target="bbb",
        top_k=3,
    )
    assert "[Ví dụ 1/3] Giải thích" in out
    assert "Câu gốc  : bbb" in out
    assert "Câu nhiễu: bbb-noised" in out
    # The marker appears at rank 1 (the gold is top-scored).
    lines = out.splitlines()
    r1_idx = next(i for i, ln in enumerate(lines) if ">> R1 <<" in ln)
    assert "[TRÚNG]" in lines[r1_idx + 1]
    assert "bbb" in lines[r1_idx + 1]


def test_format_example_block_reports_gold_outside_topk_AC1() -> None:
    docs = ["aaa", "bbb", "ccc", "ddd", "eee"]
    # target "ddd" but it's the lowest-scored -> falls outside top-3.
    row = np.array([0.9, 0.8, 0.7, 0.1, 0.6], dtype=np.float32)
    out = format_example_block(
        index_1based=2,
        n_total=2,
        explanation="x",
        clean="ddd",
        noised="ddd-noised",
        retriever_results={"R1": row},
        docs=docs,
        target="ddd",
        top_k=3,
    )
    # Top-3 list does not contain the [TRÚNG] marker; instead a "gold xếp #N"
    # follow-up line appears.
    assert out.count("[TRÚNG]") == 0
    assert "gold xếp #5" in out


def test_format_example_block_clean_sanity_label_AC1() -> None:
    docs = ["xx"]
    row = np.array([1.0], dtype=np.float32)
    out = format_example_block(
        index_1based=1,
        n_total=1,
        explanation="đối chứng",
        clean="xx",
        noised="xx",
        retriever_results={"R1": row},
        docs=docs,
        target="xx",
        top_k=1,
    )
    assert "(không có — đối chứng)" in out


def test_format_example_block_raises_when_target_missing_AC1() -> None:
    with pytest.raises(ValueError, match="not in docs index"):
        format_example_block(
            index_1based=1,
            n_total=1,
            explanation="x",
            clean="gold",
            noised="noised",
            retriever_results={"R1": np.array([1.0], dtype=np.float32)},
            docs=["other"],
            target="gold",
            top_k=1,
        )


# ---- run_canned tally semantics --------------------------------------------


def _make_examples(targets: list[str]) -> list[CannedExample]:
    return [
        CannedExample(
            id=f"ex{i}",
            vi_explanation=f"giải thích {i}",
            clean_target=t,
            mode=None,  # noised == clean, so the matrix below is deterministic
        )
        for i, t in enumerate(targets, 1)
    ]


def _row_with_gold_rank(n_docs: int, target_idx: int, rank: int) -> np.ndarray:
    """A score row in which ``target_idx`` lands at exactly ``rank`` (1-indexed).

    Places ``rank - 1`` distractor docs strictly above the gold and the rest
    strictly below; deterministic and tie-free for the gold position.
    """
    row = np.full(n_docs, 0.10, dtype=np.float32)  # below-gold filler
    row[target_idx] = 0.50
    placed = 0
    for j in range(n_docs):
        if j == target_idx:
            continue
        if placed < rank - 1:
            row[j] = 0.50 + 0.01 * (placed + 1)  # strictly above gold, distinct
            placed += 1
    return row


def test_run_canned_tallies_four_way_AC1() -> None:
    """Four examples + three retrievers, useful_k=3, hand-authored gold ranks:

    * "alpha": C1 #1, baselines #5/#6 -> CLEAN WIN (C1 in top-3, baselines not)
    * "beta":  C1 #1, baselines #1/#2 -> TIE (best baseline also #1)
    * "gamma": C1 #5, baselines #8/#9 -> GRACEFUL (none in top-3, C1 closest)
    * "delta": C1 #6, baselines #2/#3 -> LOSS (C1 worse than best baseline)
    """
    pool = [f"pool_{i}" for i in range(6)]
    examples = _make_examples(["alpha", "beta", "gamma", "delta"])
    docs = _build_index(pool, examples)  # 6 pool + 4 targets = 10 docs
    n = len(docs)
    idx = {t: docs.index(t) for t in ("alpha", "beta", "gamma", "delta")}

    # Per-retriever desired gold rank per example (order: alpha, beta, gamma, delta).
    c1_ranks = [1, 1, 5, 6]
    maxsim_ranks = [5, 1, 8, 2]
    dense_ranks = [6, 2, 9, 3]
    order = ["alpha", "beta", "gamma", "delta"]

    def make_fn(rank_list):
        def fn(queries, _docs):
            mat = np.zeros((len(queries), n), dtype=np.float32)
            for i, name in enumerate(order):
                mat[i] = _row_with_gold_rank(n, idx[name], rank_list[i])
            return mat

        return fn

    retrievers = {
        "C1": FakeRetriever(make_fn(c1_ranks)),
        "Baseline MaxSim": FakeRetriever(make_fn(maxsim_ranks)),
        "Baseline Dense": FakeRetriever(make_fn(dense_ranks)),
    }

    buf = io.StringIO()
    tally = run_canned(
        retrievers=retrievers,
        doc_pool=pool,
        examples=examples,
        c1_label="C1",
        top_k=3,
        useful_k=3,
        stream=buf,
    )
    assert tally == {"clean_wins": 1, "ties": 1, "graceful": 1, "losses": 1}

    out = buf.getvalue()
    assert "C1 THẮNG RÕ" in out
    assert "HÒA" in out
    assert "C1 TRỤ TỐT HƠN" in out
    assert "C1 THUA" in out
    assert "thắng rõ 1/4" in out
    assert "trụ tốt hơn 1/4" in out
    assert "ngưỡng hữu dụng: top-3" in out


def test_run_canned_rejects_wrong_shape_AC1() -> None:
    pool = ["x", "y"]
    examples = _make_examples(["alpha"])

    def bad(queries, docs):
        return np.zeros((len(queries) + 1, len(docs)), dtype=np.float32)

    with pytest.raises(ValueError, match="returned shape"):
        run_canned(
            retrievers={"R": FakeRetriever(bad)},
            doc_pool=pool,
            examples=examples,
            c1_label="R",
            top_k=2,
            stream=io.StringIO(),
        )


def test_run_canned_requires_examples_AC1() -> None:
    with pytest.raises(ValueError, match="no canned examples"):
        run_canned(
            retrievers={"R": FakeRetriever(lambda q, d: np.zeros((len(q), len(d))))},
            doc_pool=["a"],
            examples=[],
            stream=io.StringIO(),
        )


# ---- run_interactive --------------------------------------------------------


def test_run_interactive_exits_on_empty_query_AC1() -> None:
    """Empty stdin line -> immediate exit; no queries executed."""
    n = run_interactive(
        retrievers={"R": FakeRetriever(lambda q, d: np.zeros((len(q), len(d))))},
        doc_pool=["alpha", "beta"],
        stream=io.StringIO(),
        stream_in=io.StringIO("\n"),
    )
    assert n == 0


def test_run_interactive_runs_one_query_then_exits_AC1() -> None:
    """One query + none-mode, then empty line to exit -> 1 query executed."""

    def fake_score(queries, docs):
        mat = np.zeros((len(queries), len(docs)), dtype=np.float32)
        for i, q in enumerate(queries):
            for j, d in enumerate(docs):
                if q == d:
                    mat[i, j] = 1.0
        return mat

    # User session:
    #   "alpha\n"        -> query
    #   "none\n"         -> noise mode (no noise)
    #   "\n"             -> exit
    in_buf = io.StringIO("alpha\nnone\n\n")
    out_buf = io.StringIO()
    n = run_interactive(
        retrievers={"R": FakeRetriever(fake_score)},
        doc_pool=["pool1", "pool2"],
        stream=out_buf,
        stream_in=in_buf,
    )
    assert n == 1
    out = out_buf.getvalue()
    assert "Câu gốc  : alpha" in out
    assert "Câu nhiễu: alpha" in out
    # The query is appended to the index, so [TRÚNG] should appear at rank 1.
    assert "[TRÚNG] alpha" in out


def test_run_interactive_eof_during_mode_prompt_AC1() -> None:
    """EOF mid-session is safe (we exit cleanly)."""
    in_buf = io.StringIO("alpha\n")  # query, then EOF on mode prompt
    out_buf = io.StringIO()
    n = run_interactive(
        retrievers={"R": FakeRetriever(lambda q, d: np.zeros((len(q), len(d))))},
        doc_pool=["x"],
        stream=out_buf,
        stream_in=in_buf,
    )
    assert n == 0


# ---- _classify_verdict (honest, useful_k-aware) ----------------------------


def test_classify_verdict_clean_win_AC1() -> None:
    # C1 in top-useful_k, best baseline outside it.
    ranks = {"C1": 1, "B1": 12, "B2": 14}
    assert _classify_verdict(ranks, "C1", useful_k=10) == "clean_win"


def test_classify_verdict_graceful_AC1() -> None:
    # Neither in top-useful_k, but C1 ranks the gold far higher: graceful, not win.
    ranks = {"C1": 50, "B1": 1370, "B2": 1661}
    assert _classify_verdict(ranks, "C1", useful_k=10) == "graceful"


def test_classify_verdict_tie_when_both_useful_AC1() -> None:
    # Both surface it in top-useful_k -> tie, even though C1 has the better rank.
    ranks = {"C1": 1, "B1": 3, "B2": 8}
    assert _classify_verdict(ranks, "C1", useful_k=10) == "tie"
    # Exact rank equality is also a tie.
    assert _classify_verdict({"C1": 1, "B1": 1}, "C1", useful_k=10) == "tie"


def test_classify_verdict_loss_AC1() -> None:
    ranks = {"C1": 7, "B1": 2, "B2": 9}
    assert _classify_verdict(ranks, "C1", useful_k=10) == "loss"


def test_classify_verdict_useful_k_boundary_AC1() -> None:
    # C1 exactly at useful_k counts as useful; baseline one past it does not.
    assert _classify_verdict({"C1": 10, "B1": 11}, "C1", useful_k=10) == "clean_win"
    # Both just past the boundary -> graceful.
    assert _classify_verdict({"C1": 11, "B1": 12}, "C1", useful_k=10) == "graceful"


# ---- _recommend_seed --------------------------------------------------------


def test_recommend_seed_prefers_clean_win_with_widest_margin_AC1() -> None:
    # seed 2 and 3 are clean wins (C1 in top-10, baseline out); 3 has wider margin.
    per_seed = [
        (0, 1, {"b": 2}),  # tie (both useful)
        (1, 50, {"b": 1370}),  # graceful
        (2, 3, {"b": 40}),  # clean win, margin 37
        (3, 2, {"b": 900}),  # clean win, margin 898  <- best
    ]
    assert _recommend_seed(per_seed, useful_k=10) == 3


def test_recommend_seed_falls_back_to_graceful_then_best_rank_AC1() -> None:
    # No clean win; pick the graceful seed with the smallest C1 rank.
    per_seed = [
        (0, 80, {"b": 90}),  # graceful
        (1, 40, {"b": 120}),  # graceful, better C1 rank  <- best
    ]
    assert _recommend_seed(per_seed, useful_k=10) == 1
    # No clean win and no graceful (all ties/losses): smallest C1 rank.
    per_seed2 = [(0, 1, {"b": 1}), (1, 2, {"b": 1})]
    assert _recommend_seed(per_seed2, useful_k=10) == 0


# ---- tune_seeds -------------------------------------------------------------


def test_tune_seeds_structure_and_skips_control_AC1() -> None:
    """tune_seeds returns a recommended seed per non-control example, in range,
    skips the clean-sanity control, and prints a recommendation line."""
    examples = [
        CannedExample(id="n1", vi_explanation="x", clean_target="alpha", mode=NoiseMode.DROP_ALL),
        CannedExample(id="n2", vi_explanation="x", clean_target="beta", mode=NoiseMode.MIXED_OCR),
        CannedExample(id="ctl", vi_explanation="x", clean_target="gamma", mode=None),
    ]

    buf = io.StringIO()
    best = tune_seeds(
        retrievers={
            "C1": FakeRetriever(lambda q, d: np.zeros((len(q), len(d)))),
            "B1": FakeRetriever(lambda q, d: np.zeros((len(q), len(d)))),
        },
        doc_pool=["p0", "p1", "p2"],
        examples=examples,
        c1_label="C1",
        sweep_n=4,
        useful_k=10,
        stream=buf,
    )
    assert set(best.keys()) == {"n1", "n2"}  # control skipped
    assert all(0 <= s < 4 for s in best.values())
    assert "recommend noise_seed=" in buf.getvalue()
    assert "[Đề xuất seed]" in buf.getvalue()
