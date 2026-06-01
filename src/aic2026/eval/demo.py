# Implements SPEC-0014 SS 6 (live demo for C1 DiacriticBERT).
"""Live demo for the C1 DiacriticBERT ship-gate.

Two surfaces:

  * :func:`run_canned` -- iterates over :data:`CANNED_EXAMPLES`, each of which
    is a hand-picked Vietnamese clean target paired with the ``NoiseMode`` that
    best exhibits the head's contribution. For each example we noise the clean
    target with the *same* deterministic ``noise()`` used at training time,
    score the noised query against every retriever, and print a top-k block
    that marks where the gold target landed. Ends with a win/tie/loss tally.

  * :func:`run_interactive` -- a REPL. The audience types a Vietnamese query,
    optionally picks a noise mode (or types a custom noised form), and sees
    the same side-by-side block. Useful for Q&A after the canned showcase.

The retrievers are passed in as a ``{display_label: Retriever}`` mapping so
the caller (CLI) owns the BGE-M3 wiring + chooses Vietnamese display names.
The demo module is pure and torch-free in its imports: the retrievers may be
``DummyEmbedder``-backed for tests, or real ``MaxSimRetriever`` /
``BgeM3DenseEmbedder`` on a GPU box for the live run.

Docs index: the demo combines an external ``doc_pool`` (typically a few
hundred clean Vietnamese strings from :func:`build_heldout_queries`) with the
canned ``clean_target`` strings so the gold answer is always present in the
index. The combined list is deduped while preserving insertion order.
"""

from __future__ import annotations

import dataclasses
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import IO, TYPE_CHECKING

import numpy as np

from aic2026.train.diacritic_noise import NoiseMode, noise

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from aic2026.eval.retrievers import Retriever


C1_LABEL_DEFAULT = "C1 (head đã huấn luyện)"
"""Default Vietnamese display label for the C1 retriever. The CLI passes the
same label as a key in the ``retrievers`` mapping; the win/tie/loss tally uses
``c1_label`` to identify which row is C1."""


@dataclasses.dataclass(frozen=True)
class CannedExample:
    """One canned demo example.

    ``clean_target`` is what we expect the retriever to surface; it's inserted
    into the index by :func:`run_canned`. ``mode`` is the noise mode applied to
    produce the query; ``None`` means "no noise" (a sanity-check example all
    three retrievers should land top-1). ``noise_seed`` keys the per-example
    RNG so the noised string is byte-stable across runs.
    """

    id: str
    vi_explanation: str
    clean_target: str
    mode: NoiseMode | None
    noise_seed: int = 0

    def make_noised(self) -> str:
        """Return the noised form of ``clean_target`` (or the clean target if
        ``mode is None``). Deterministic per ``(id, noise_seed)``."""
        if self.mode is None:
            return self.clean_target
        rng = random.Random(f"{self.noise_seed}\x00demo\x00{self.id}")
        return noise(self.clean_target, self.mode, rng=rng)


CANNED_EXAMPLES: tuple[CannedExample, ...] = (
    # --- Diacritic loss (drop_all) -------------------------------------------
    CannedExample(
        id="ex1_drop_all_placename",
        vi_explanation=(
            "Mất toàn bộ dấu — gặp khi OCR đọc biển hiệu cũ hoặc font không hỗ trợ tiếng Việt"
        ),
        clean_target="Hà Nội là thủ đô của Việt Nam",
        mode=NoiseMode.DROP_ALL,
    ),
    CannedExample(
        id="ex10_drop_all_heritage",
        vi_explanation=(
            "Mất dấu trên câu dài — chú ý từ nước ngoài (UNESCO) vốn không có dấu nên không đổi"
        ),
        clean_target="Vịnh Hạ Long là di sản thiên nhiên thế giới được UNESCO công nhận",
        mode=NoiseMode.DROP_ALL,
    ),
    # --- Visual confusables (char_confuse) -----------------------------------
    CannedExample(
        id="ex2_char_confuse_address",
        vi_explanation=(
            "Nhầm chữ số với chữ cái (5↔S, 1↔l, 0↔O) — phổ biến trên biển số nhà và địa chỉ"
        ),
        clean_target="Số 5 đường Lê Lợi, Quận 1",
        mode=NoiseMode.CHAR_CONFUSE,
    ),
    CannedExample(
        id="ex11_char_confuse_news",
        vi_explanation=(
            "Nhầm ký tự trên tỉ số thể thao (3→E, 0→o) — OCR đọc sai số liệu trong khung hình"
        ),
        clean_target="Đội tuyển Việt Nam thắng 3 0 trước Thái Lan",
        mode=NoiseMode.CHAR_CONFUSE,
    ),
    # --- Segmentation + tonal (word_merge / homophone_swap) ------------------
    CannedExample(
        id="ex4_word_merge_phrase",
        vi_explanation=(
            "Mất khoảng trắng giữa các từ (under-segmentation) — OCR cleaner gộp các từ liền nhau"
        ),
        clean_target="quả táo đỏ trên bàn",
        mode=NoiseMode.WORD_MERGE,
        noise_seed=6,
    ),
    CannedExample(
        id="ex12_homophone_singer",
        vi_explanation=(
            "Lỗi đồng-âm-khác-dấu của ASR (PhoWhisper): 'Mỹ Tâm' → 'Mỹ Tấm' — "
            "đây là lỗi tone_swap không bắt được vì nó neo theo dấu sẵn có"
        ),
        clean_target="Ca sĩ Mỹ Tâm biểu diễn tại nhà hát lớn Hà Nội",
        mode=NoiseMode.HOMOPHONE_SWAP,
        noise_seed=2,
    ),
    # --- Realistic combined OCR noise (mixed_ocr) — the mode C1 wins on -------
    CannedExample(
        id="ex3_mixed_ocr_long",
        vi_explanation=(
            "Tổ hợp nhiễu OCR thực tế (mất dấu + nhầm ký tự + tách chữ) — "
            "C1 thắng nhiều nhất ở chế độ này (eval v3: +6.4pp so với v2)"
        ),
        clean_target="Cuộc thi AI Challenge sẽ tổ chức tại Hà Nội vào tháng 7",
        mode=NoiseMode.MIXED_OCR,
    ),
    CannedExample(
        id="ex6_mixed_ocr_gold_price",
        vi_explanation=("Nhiễu OCR trên câu tin tức có số liệu (75 → Ts) — cả chữ lẫn số đều hỏng"),
        clean_target="Giá vàng trong nước tăng lên 75 triệu đồng một lượng",
        mode=NoiseMode.MIXED_OCR,
        noise_seed=5,
    ),
    CannedExample(
        id="ex7_mixed_ocr_lifelog",
        vi_explanation=(
            "Nhiễu OCR trên câu mô tả cảnh (truy vấn lifelog điển hình) — "
            "địa danh 'Long Biên' bị phá thành 'L0nq 8ieh'"
        ),
        clean_target="Người phụ nữ mặc áo dài trắng đi trên cầu Long Biên",
        mode=NoiseMode.MIXED_OCR,
        noise_seed=5,
    ),
    CannedExample(
        id="ex8_mixed_ocr_stadium",
        vi_explanation=(
            "Nhiễu OCR trên câu nhiều địa danh — tên riêng 'Mỹ Đình', 'Nam Từ Liêm' khó phục hồi"
        ),
        clean_target="Sân vận động quốc gia Mỹ Đình nằm ở quận Nam Từ Liêm",
        mode=NoiseMode.MIXED_OCR,
        noise_seed=2,
    ),
    CannedExample(
        id="ex9_mixed_ocr_flight",
        vi_explanation=(
            "Nhiễu OCR trên mã chuyến bay (1546 → IsAG) — mã chữ-số là trường hợp tệ nhất cho OCR"
        ),
        clean_target="Chuyến bay VN 1546 khởi hành lúc 8 giờ sáng tại Tân Sơn Nhất",
        mode=NoiseMode.MIXED_OCR,
    ),
    # --- Control (no noise) — keep last --------------------------------------
    CannedExample(
        id="ex5_clean_sanity",
        vi_explanation=(
            "Đối chứng — câu sạch không nhiễu: cả ba bộ truy hồi đều phải "
            "đứng top-1 (chứng minh C1 không làm hỏng truy vấn sạch)"
        ),
        clean_target="Trường Đại học Bách khoa Hà Nội",
        mode=None,
    ),
)


# ---- index building ---------------------------------------------------------


def _build_index(doc_pool: Iterable[str], canned: Iterable[CannedExample]) -> list[str]:
    """Dedupe ``doc_pool`` (insertion order), then append any canned target
    not already present. The gold target for every canned example is therefore
    guaranteed to be in the returned index."""
    seen: set[str] = set()
    docs: list[str] = []
    for d in doc_pool:
        if d and d not in seen:
            seen.add(d)
            docs.append(d)
    for ex in canned:
        if ex.clean_target not in seen:
            seen.add(ex.clean_target)
            docs.append(ex.clean_target)
    return docs


# ---- formatting -------------------------------------------------------------


def _rank_of(score_row: np.ndarray, target_idx: int) -> int:
    """1-indexed position of ``target_idx`` in ``score_row`` sorted desc."""
    order = np.argsort(-score_row)
    return int(np.where(order == target_idx)[0][0]) + 1


def _render_one_retriever(
    *, label: str, row: np.ndarray, docs: Sequence[str], target_idx: int, top_k: int
) -> list[str]:
    """Render one retriever's top-k block as a list of lines."""
    order = np.argsort(-row)[:top_k]
    lines: list[str] = [f"  >> {label} <<"]
    for rank, di in enumerate(order, 1):
        marker = "[TRÚNG]" if int(di) == target_idx else "       "
        lines.append(f"    {rank}. {marker} {docs[int(di)]}    ({float(row[int(di)]):+.3f})")
    if target_idx not in order:
        lines.append(f"        (gold xếp #{_rank_of(row, target_idx)} ngoài top-{top_k})")
    lines.append("")
    return lines


def format_example_block(
    *,
    index_1based: int,
    n_total: int,
    explanation: str,
    clean: str,
    noised: str,
    retriever_results: Mapping[str, np.ndarray],
    docs: Sequence[str],
    target: str,
    top_k: int = 3,
) -> str:
    """Render one canned example as a side-by-side comparison block.

    ``retriever_results`` maps display label -> ``(nd,)`` score row. Insertion
    order of the mapping is preserved in the output (Python 3.7+ guarantee).
    """
    if target not in docs:
        raise ValueError(f"target {target!r} not in docs index")
    target_idx = docs.index(target)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"[Ví dụ {index_1based}/{n_total}] {explanation}")
    lines.append("-" * 72)
    lines.append(f"  Câu gốc  : {clean}")
    if noised != clean:
        lines.append(f"  Câu nhiễu: {noised}")
    else:
        lines.append("  Câu nhiễu: (không có — đối chứng)")
    lines.append("")
    for label, row in retriever_results.items():
        lines.extend(
            _render_one_retriever(
                label=label, row=row, docs=docs, target_idx=target_idx, top_k=top_k
            )
        )
    return "\n".join(lines) + "\n"


# ---- canned runner ----------------------------------------------------------


def _tally_verdict(ranks: Mapping[str, int], c1_label: str) -> str:
    """Return 'win' / 'tie' / 'loss' for C1 relative to the best baseline."""
    if c1_label not in ranks:
        return "n/a"
    baseline_ranks = [r for lbl, r in ranks.items() if lbl != c1_label]
    if not baseline_ranks:
        return "n/a"
    best_baseline = min(baseline_ranks)
    if ranks[c1_label] < best_baseline:
        return "win"
    if ranks[c1_label] == best_baseline:
        return "tie"
    return "loss"


def run_canned(
    *,
    retrievers: Mapping[str, Retriever],
    doc_pool: Sequence[str],
    examples: Sequence[CannedExample] = CANNED_EXAMPLES,
    c1_label: str = C1_LABEL_DEFAULT,
    top_k: int = 3,
    stream: IO[str] = sys.stdout,
) -> dict[str, int]:
    """Run all canned examples; return the win/tie/loss tally as a dict.

    The retrievers are batched: each retriever sees all queries in one call
    (3 retrievers x 1 doc-encoding each, not per-example). On an H200 this
    completes in ~20s for ~300-doc indexes.
    """
    if not examples:
        raise ValueError("no canned examples provided")
    docs = _build_index(doc_pool, examples)
    queries = [ex.make_noised() for ex in examples]

    per_retriever_scores: dict[str, np.ndarray] = {}
    for label, r in retrievers.items():
        scores = np.asarray(r.score(queries, docs), dtype=np.float32)
        if scores.shape != (len(queries), len(docs)):
            raise ValueError(
                f"retriever {label!r} returned shape {scores.shape}; "
                f"expected ({len(queries)}, {len(docs)})"
            )
        per_retriever_scores[label] = scores

    tally = {"wins": 0, "ties": 0, "losses": 0}
    for i, ex in enumerate(examples, 1):
        noised = queries[i - 1]
        row_per = {lbl: per_retriever_scores[lbl][i - 1] for lbl in retrievers}
        block = format_example_block(
            index_1based=i,
            n_total=len(examples),
            explanation=ex.vi_explanation,
            clean=ex.clean_target,
            noised=noised,
            retriever_results=row_per,
            docs=docs,
            target=ex.clean_target,
            top_k=top_k,
        )
        stream.write("\n" + block)

        target_idx = docs.index(ex.clean_target)
        ranks = {lbl: _rank_of(row, target_idx) for lbl, row in row_per.items()}
        verdict = _tally_verdict(ranks, c1_label)
        if verdict == "win":
            tally["wins"] += 1
            verdict_vi = "C1 THẮNG"
        elif verdict == "tie":
            tally["ties"] += 1
            verdict_vi = "HÒA"
        elif verdict == "loss":
            tally["losses"] += 1
            verdict_vi = "C1 THUA"
        else:
            verdict_vi = "n/a"
        ranks_str = "  ".join(f"{lbl}=#{r}" for lbl, r in ranks.items())
        stream.write(f"  >>> Phán quyết: {verdict_vi}  ({ranks_str})\n")

    n = len(examples)
    stream.write("\n" + "=" * 72 + "\n")
    stream.write(
        f"[Tổng kết] C1 thắng {tally['wins']}/{n}  "
        f"hòa {tally['ties']}/{n}  thua {tally['losses']}/{n}\n"
    )
    return tally


# ---- interactive REPL -------------------------------------------------------


_NOISE_MODES_VI = ", ".join(m.value for m in NoiseMode)
_INTERACTIVE_BANNER = (
    "\n"
    + "=" * 72
    + "\nChế độ tương tác — gõ truy vấn, nhấn Enter rỗng để thoát.\n"
    + "Các chế độ nhiễu: "
    + _NOISE_MODES_VI
    + ", none, custom\n"
    + "=" * 72
    + "\n"
)


def _read_line(stream_in: IO[str]) -> str | None:
    """Read one line or return None on EOF."""
    line = stream_in.readline()
    if not line:  # truly EOF
        return None
    return line.rstrip("\n").rstrip("\r")


def _resolve_noise(clean: str, mode_input: str, stream: IO[str], stream_in: IO[str]) -> str | None:
    """Translate the mode-input string into a noised query. Returns ``None``
    only if the user aborts during the ``custom`` prompt."""
    mode_input = mode_input.strip().lower()
    if mode_input == "custom":
        stream.write("> Câu nhiễu (gõ trực tiếp): ")
        stream.flush()
        custom = _read_line(stream_in)
        return None if custom is None else custom
    if mode_input in ("", "none"):
        return clean
    try:
        mode = NoiseMode(mode_input)
    except ValueError:
        stream.write(f"(không nhận diện chế độ {mode_input!r}; dùng câu sạch)\n")
        return clean
    rng = random.Random(f"interactive\x00{clean}\x00{mode_input}")
    return noise(clean, mode, rng=rng)


def run_interactive(
    *,
    retrievers: Mapping[str, Retriever],
    doc_pool: Sequence[str],
    top_k: int = 5,
    stream: IO[str] = sys.stdout,
    stream_in: IO[str] = sys.stdin,
) -> int:
    """REPL. Returns the number of queries executed (0 if the user exited
    immediately). Index is built once from ``doc_pool`` (deduped); user queries
    are appended to the index on the fly so the user can spot their own input."""
    docs = _build_index(doc_pool, [])
    docs_seen: set[str] = set(docs)
    stream.write(_INTERACTIVE_BANNER)
    stream.write(f"Chỉ mục ban đầu có {len(docs)} câu tiếng Việt.\n")

    n_queries = 0
    while True:
        stream.write("\n> Câu truy vấn (rỗng để thoát): ")
        stream.flush()
        clean = _read_line(stream_in)
        if clean is None or not clean.strip():
            break
        clean = clean.strip()

        stream.write("> Chế độ nhiễu (mặc định none): ")
        stream.flush()
        mode_input = _read_line(stream_in)
        if mode_input is None:
            break
        noised = _resolve_noise(clean, mode_input, stream, stream_in)
        if noised is None:
            break

        if clean not in docs_seen:
            docs.append(clean)
            docs_seen.add(clean)

        target_idx = docs.index(clean)
        stream.write(f"\n  Câu gốc  : {clean}\n")
        stream.write(f"  Câu nhiễu: {noised}\n\n")
        for label, r in retrievers.items():
            row = np.asarray(r.score([noised], docs), dtype=np.float32)[0]
            stream.write(
                "\n".join(
                    _render_one_retriever(
                        label=label, row=row, docs=docs, target_idx=target_idx, top_k=top_k
                    )
                )
                + "\n"
            )
        n_queries += 1
    return n_queries
