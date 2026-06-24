# Implements SPEC-0014 Q2 (noise-schedule realism calibration on real corpus text).
"""Calibrate the C1 synthetic noise schedule against real corpus text.

[SPEC-0014](../../../docs/specs/SPEC-0014-diacritic-bert.md) Q2 is open: the
`Beta(2, 5)` noise severity and the choice of modes are *assumptions* that should
be validated against real PhoWhisper/PaddleOCR output. Before the AIC2025 proxy
corpus ([research-note 07](../../../docs/research-notes/07-aic2025-proxy-corpus.md))
we had no real Vietnamese OCR/ASR text to check against; now we do.

We can't compute a true OCR *error rate* (that needs aligned clean/noisy pairs we
don't have), so we calibrate by **distribution matching of surface statistics**:

  * Does our **clean anchor** distribution (currently Wikipedia) resemble the
    **real query** distribution (length, diacritic density)? If not, bias anchor
    selection toward query-like text.
  * Does our **synthetic noised** output (esp. `mixed_ocr`) resemble **real OCR
    output** in fragmentation (single-char-token ratio), digit density, and
    casing? If real OCR is far more/less fragmented, retune `space_split` /
    `char_confuse` probabilities.

Everything here is pure stdlib + the noise module, so it imports and tests on
CPU/CI without torch/pyarrow. The CLI (`bin/train c1-calibrate`) lazy-loads
the parquet reader only when `--pairs` is given.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
import statistics
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from aic2026.train.diacritic_noise import NoiseMode, noise

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TextStats:
    """Cheap surface statistics over a list of strings (averaged per-string).

    All ratios are means of the per-string ratio, so short and long strings are
    weighted equally - we care about the *shape* of a typical string, not the
    corpus token totals.
    """

    n: int
    char_len_mean: float
    char_len_median: float
    word_len_mean: float
    diacritic_ratio_mean: float  # Vietnamese diacritic-bearing alpha fraction
    digit_ratio_mean: float  # digits / chars
    upper_ratio_mean: float  # uppercase / alpha
    space_ratio_mean: float  # spaces / chars (raw fragmentation)
    single_char_token_ratio_mean: float  # len-1 tokens / tokens (OCR over-seg tell)

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def _diacritic_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    marked = 0
    for ch in alpha:
        if ch in ("đ", "Đ") or any(
            unicodedata.combining(d) for d in unicodedata.normalize("NFD", ch)
        ):
            marked += 1
    return marked / len(alpha)


def _ratio(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def _single_char_token_ratio(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if len(t) == 1) / len(toks)


def profile_text(strings: Sequence[str]) -> TextStats:
    """Compute :class:`TextStats` over ``strings`` (non-empty after strip)."""
    items = [s for s in strings if s and s.strip()]
    if not items:
        raise ValueError("profile_text got no non-empty strings")

    char_lens = [len(s) for s in items]
    word_lens = [len(s.split()) for s in items]
    dia = [_diacritic_ratio(s) for s in items]
    digit = [_ratio(sum(c.isdigit() for c in s), len(s)) for s in items]
    upper = [_ratio(sum(c.isupper() for c in s), sum(c.isalpha() for c in s)) for s in items]
    space = [_ratio(s.count(" "), len(s)) for s in items]
    single = [_single_char_token_ratio(s) for s in items]

    return TextStats(
        n=len(items),
        char_len_mean=round(statistics.mean(char_lens), 2),
        char_len_median=round(float(statistics.median(char_lens)), 2),
        word_len_mean=round(statistics.mean(word_lens), 2),
        diacritic_ratio_mean=round(statistics.mean(dia), 4),
        digit_ratio_mean=round(statistics.mean(digit), 4),
        upper_ratio_mean=round(statistics.mean(upper), 4),
        space_ratio_mean=round(statistics.mean(space), 4),
        single_char_token_ratio_mean=round(statistics.mean(single), 4),
    )


def profile_synthetic_noise(
    anchors: Sequence[str],
    *,
    modes: Sequence[NoiseMode] = tuple(NoiseMode),
    seed: int = 0,
    max_anchors: int | None = 2000,
) -> dict[str, TextStats]:
    """Apply each noise ``mode`` to ``anchors`` and profile the output per mode."""
    pool = list(anchors)
    if max_anchors is not None and len(pool) > max_anchors:
        rng = random.Random(f"calib\x00{seed}")
        pool = rng.sample(pool, max_anchors)
    out: dict[str, TextStats] = {}
    for mode in modes:
        noised = [noise(a, mode, rng=random.Random(f"{seed}\x00{mode.value}\x00{a}")) for a in pool]
        out[mode.value] = profile_text(noised)
    return out


# --- calibration verdict ------------------------------------------------------

_LEN_RATIO_FLAG = 2.0  # anchor vs query mean-length factor that triggers a flag
_FRAG_ABS_FLAG = 0.10  # abs gap in single-char-token ratio that triggers a flag


def _flag_anchor_vs_query(anchor: TextStats, query: TextStats) -> list[str]:
    flags: list[str] = []
    a, q = anchor.char_len_mean, query.char_len_mean
    if q > 0 and (a / q >= _LEN_RATIO_FLAG or q / a >= _LEN_RATIO_FLAG):
        longer = "longer" if a > q else "shorter"
        flags.append(
            f"anchor length mismatch: our anchors are ~{a:.0f} chars vs real queries "
            f"~{q:.0f} ({longer}); bias anchor selection toward query-like length"
        )
    if abs(anchor.diacritic_ratio_mean - query.diacritic_ratio_mean) >= 0.10:
        flags.append(
            f"diacritic density mismatch: anchors {anchor.diacritic_ratio_mean:.2f} "
            f"vs queries {query.diacritic_ratio_mean:.2f}"
        )
    return flags


def _flag_synthetic_vs_ocr(synthetic_mixed: TextStats, ocr: TextStats) -> list[str]:
    flags: list[str] = []
    d = synthetic_mixed.single_char_token_ratio_mean - ocr.single_char_token_ratio_mean
    if abs(d) >= _FRAG_ABS_FLAG:
        direction = "over-fragments" if d > 0 else "under-fragments"
        flags.append(
            f"mixed_ocr {direction} vs real OCR (single-char-token ratio "
            f"{synthetic_mixed.single_char_token_ratio_mean:.2f} vs "
            f"{ocr.single_char_token_ratio_mean:.2f}); retune space_split probability"
        )
    if abs(synthetic_mixed.digit_ratio_mean - ocr.digit_ratio_mean) >= 0.05:
        flags.append(
            f"digit density mismatch: mixed_ocr {synthetic_mixed.digit_ratio_mean:.3f} "
            f"vs real OCR {ocr.digit_ratio_mean:.3f}"
        )
    return flags


def compare(
    *,
    real_query: TextStats | None,
    our_anchor: TextStats | None,
    real_ocr: TextStats | None,
    synthetic: dict[str, TextStats] | None,
) -> dict[str, object]:
    """Assemble a calibration report + advisory flags from the profiled sources.

    Any source may be ``None`` (not provided). Flags are heuristic and advisory:
    they point at the noise knob to revisit, they do not auto-tune it.
    """
    flags: list[str] = []
    if our_anchor is not None and real_query is not None:
        flags += _flag_anchor_vs_query(our_anchor, real_query)
    if synthetic is not None and real_ocr is not None and "mixed_ocr" in synthetic:
        flags += _flag_synthetic_vs_ocr(synthetic["mixed_ocr"], real_ocr)

    return {
        "real_query": real_query.as_dict() if real_query else None,
        "our_anchor": our_anchor.as_dict() if our_anchor else None,
        "real_ocr": real_ocr.as_dict() if real_ocr else None,
        "synthetic": {m: s.as_dict() for m, s in synthetic.items()} if synthetic else None,
        "flags": flags,
        "verdict": "calibration OK (no flags)"
        if not flags
        else f"{len(flags)} flag(s) - see flags[]",
    }


# --- text loading (CLI helper) ------------------------------------------------


def load_strings(path: Path) -> list[str]:
    """Load query/OCR strings from a file or directory.

    Handles ``.txt``/``.tsv``/``.csv`` (one string per non-empty line),
    ``.json``/``.jsonl`` (string leaves of length >= 4), and ``.xlsx``/``.xls``
    (string cell values of length >= 4, read via an optional ``openpyxl``
    import). A directory is walked for any of those extensions. Best-effort and
    defensive - unknown formats yield ``[]`` rather than raising, and
    spreadsheets are skipped with a logged warning when ``openpyxl`` is absent.
    """
    exts = {".txt", ".tsv", ".csv", ".json", ".jsonl", ".xlsx", ".xls"}
    files = (
        [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]
        if path.is_dir()
        else [path]
    )
    out: list[str] = []
    for p in files:
        suffix = p.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            out.extend(_xlsx_strings(p))
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if suffix in (".json", ".jsonl"):
            chunks = raw.splitlines() if suffix == ".jsonl" else [raw]
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    out.extend(_json_strings(json.loads(chunk)))
                except json.JSONDecodeError:
                    continue
        else:
            out.extend(ln.strip() for ln in raw.splitlines() if ln.strip())
    return out


def _xlsx_strings(path: Path) -> list[str]:
    """Collect string cell values (stripped, len >= 4) from every worksheet.

    ``openpyxl`` is imported lazily inside this branch so the module keeps
    importing on CPU/CI without the dependency. If it is unavailable the
    spreadsheet is skipped with a logged warning (best-effort contract), and a
    corrupt/legacy file that ``openpyxl`` cannot open is likewise skipped.
    """
    try:
        import openpyxl
    except ImportError:
        logger.warning(
            "openpyxl not installed; skipping spreadsheet %s "
            "(install openpyxl, e.g. run `uv run --with openpyxl ...`)",
            path,
        )
        return []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        logger.warning("could not read spreadsheet %s: %s", path, exc)
        return []
    found: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for value in row:
                if isinstance(value, str):
                    text = value.strip()
                    if len(text) >= 4:
                        found.append(text)
    wb.close()
    return found


def _json_strings(obj: object) -> list[str]:
    found: list[str] = []
    if isinstance(obj, str):
        if len(obj) >= 4:
            found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_json_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_json_strings(v))
    return found
