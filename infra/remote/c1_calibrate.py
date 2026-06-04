#!/usr/bin/env python
# Implements SPEC-0014 calibration support (C1 noise-schedule calibration).
# Uncommitted infra helper (NOT part of the shipped package). It REUSES the
# committed noise functions read-only (aic2026.train.diacritic_noise) - it does
# NOT modify the frozen v3 schedule. It:
#   1. computes the empirical surface-statistics of real OCR text (from
#      ocr_sample.py's JSONL),
#   2. computes the same statistics for each synthetic NoiseMode applied to a set
#      of clean Vietnamese anchors,
#   3. scores how close each mode's corruption signature is to the real-OCR
#      signature and derives a calibration-nudged per-mode weight distribution,
#   4. writes a calibration_report.json, and
#   5. emits a calibration-nudged training corpus parquet (same schema as
#      train.diacritic_corpus) so `train c1-fit` can retrain on it.
#
#   uv run python infra/remote/c1_calibrate.py \
#       --ocr /tmp/c1cal/ocr_sample.jsonl \
#       --report /tmp/c1cal/calibration_report.json \
#       --corpus-out /tmp/c1cal/pairs_calibrated.parquet \
#       --clean-from /tmp/c1/pairs.parquet
from __future__ import annotations

import argparse
import json
import math
import random
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from aic2026.train.diacritic_corpus import _mine_hard_negatives, build_corpus, read_pairs
from aic2026.train.diacritic_noise import NoiseMode, noise

# Vietnamese vowels carrying tone (mirror of diacritic_noise; read-only here).
_VOWELS = set("aeiouyAEIOUY\u0103\u00e2\u00ea\u00f4\u01a1\u01b0\u0102\u00c2\u00ca\u00d4\u01a0\u01af")
_TONE_MARKS = ("\u0300", "\u0301", "\u0303", "\u0309", "\u0323")
_TONE_SET = set(_TONE_MARKS)
_STROKE_D = {"\u0111", "\u0110"}

# The surface statistics we calibrate on. Each maps a string -> float in [0, 1]
# (or a small positive scale for mean_token_len, which we min-max later).
STAT_KEYS = (
    "diacritic_density",
    "tone_density",
    "digit_fraction",
    "space_fraction",
    "single_char_token_frac",
    "upper_frac",
    "nonascii_frac",
    "mean_token_len",
)


def _stats(text: str) -> dict[str, float]:
    """Compute the surface-statistic vector for one string."""
    if not text:
        return {k: 0.0 for k in STAT_KEYS}
    nfd = unicodedata.normalize("NFD", text)
    n_chars = len(text)
    alpha = [c for c in text if c.isalpha()]
    n_alpha = len(alpha) or 1
    combining = [c for c in nfd if unicodedata.combining(c)]
    n_diacritic = len(combining) + sum(1 for c in text if c in _STROKE_D)
    n_tone = sum(1 for c in nfd if c in _TONE_SET)
    n_vowels = sum(1 for c in text if c in _VOWELS) or 1
    n_digit = sum(1 for c in text if c.isdigit())
    n_space = sum(1 for c in text if c == " ")
    n_upper = sum(1 for c in alpha if c.isupper())
    n_nonascii = sum(1 for c in text if ord(c) > 127)
    tokens = [t for t in text.split() if t]
    n_tokens = len(tokens) or 1
    single = sum(1 for t in tokens if len(t) == 1)
    mean_tok = sum(len(t) for t in tokens) / n_tokens
    return {
        "diacritic_density": n_diacritic / n_alpha,
        "tone_density": n_tone / n_vowels,
        "digit_fraction": n_digit / n_chars,
        "space_fraction": n_space / n_chars,
        "single_char_token_frac": single / n_tokens,
        "upper_frac": n_upper / n_alpha,
        "nonascii_frac": n_nonascii / n_chars,
        "mean_token_len": mean_tok,
    }


def _agg(stats_list: Sequence[dict[str, float]]) -> dict[str, float]:
    if not stats_list:
        return {k: 0.0 for k in STAT_KEYS}
    return {k: sum(s[k] for s in stats_list) / len(stats_list) for k in STAT_KEYS}


def _load_ocr_texts(path: Path) -> list[str]:
    out: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rec = json.loads(ln)
        t = rec.get("text", "")
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out


def _clean_anchors(clean_from: Path | None, max_anchors: int) -> list[str]:
    """Clean Vietnamese anchors: reuse a cached corpus parquet if present, else
    harvest a small set via the committed build_corpus."""
    if clean_from and clean_from.exists():
        rows = read_pairs(clean_from)
        seen: set[str] = set()
        anchors: list[str] = []
        for r in rows:
            a = r.get("anchor_clean")
            if isinstance(a, str) and a not in seen:
                seen.add(a)
                anchors.append(a)
            if len(anchors) >= max_anchors:
                break
        if anchors:
            return anchors
    # fallback: harvest fresh (network + train extra)
    tmp = Path("/tmp/c1cal/_clean_harvest.parquet")
    res = build_corpus(out=tmp, k=1, max_per_source=max_anchors, seed=0)
    rows = read_pairs(res.out)
    seen2: set[str] = set()
    anchors2: list[str] = []
    for r in rows:
        a = r.get("anchor_clean")
        if isinstance(a, str) and a not in seen2:
            seen2.add(a)
            anchors2.append(a)
    return anchors2[:max_anchors]


def _mode_signatures(anchors: Sequence[str], seed: int) -> dict[str, dict[str, float]]:
    """Aggregate surface stats per NoiseMode (applied to the clean anchors)."""
    sigs: dict[str, dict[str, float]] = {}
    for mode in NoiseMode:
        per: list[dict[str, float]] = []
        for a in anchors:
            rng = random.Random(f"{seed}\x00{mode.value}\x00{a}")
            per.append(_stats(noise(a, mode, rng=rng)))
        sigs[mode.value] = _agg(per)
    return sigs


def _normalize_deltas(
    real: dict[str, float], clean: dict[str, float], scales: dict[str, float]
) -> dict[str, float]:
    """Direction+magnitude of real-vs-clean per stat, scaled to comparable units."""
    return {k: (real[k] - clean[k]) / scales[k] for k in STAT_KEYS}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a[k] * b[k] for k in STAT_KEYS)
    na = math.sqrt(sum(a[k] ** 2 for k in STAT_KEYS))
    nb = math.sqrt(sum(b[k] ** 2 for k in STAT_KEYS))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _softmax(scores: dict[str, float], temp: float) -> dict[str, float]:
    mx = max(scores.values())
    exps = {k: math.exp((v - mx) / temp) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def _weighted_modes(weights: dict[str, float], k: int, rng: random.Random) -> list[NoiseMode]:
    """Sample k modes from the weighted distribution (calibration nudge)."""
    modes = list(weights.keys())
    probs = [weights[m] for m in modes]
    chosen = rng.choices(modes, weights=probs, k=k)
    return [NoiseMode(m) for m in chosen]


def _write_corpus(
    anchors: Sequence[str],
    weights: dict[str, float],
    out: Path,
    k_per_anchor: int,
    hard_negatives: int,
    seed: int,
) -> int:
    """Emit a calibration-nudged corpus parquet (train.diacritic_corpus schema)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng_neg = random.Random(f"{seed}\x00hardnegs")
    hard = _mine_hard_negatives(anchors, n=hard_negatives, rng=rng_neg)

    a_col: list[str] = []
    p_col: list[str] = []
    m_col: list[str] = []
    n_col: list[list[str]] = []
    for anchor, negs in zip(anchors, hard, strict=True):
        sel_rng = random.Random(f"{seed}\x00sel\x00{anchor}")
        for mode in _weighted_modes(weights, k_per_anchor, sel_rng):
            nz_rng = random.Random(f"{seed}\x00{mode.value}\x00{anchor}")
            a_col.append(anchor)
            p_col.append(noise(anchor, mode, rng=nz_rng))
            m_col.append(mode.value)
            n_col.append(negs)

    table = pa.table(
        {
            "anchor_clean": pa.array(a_col, pa.string()),
            "positive_noisy": pa.array(p_col, pa.string()),
            "mode": pa.array(m_col, pa.string()),
            "hard_negs": pa.array(n_col, pa.list_(pa.string())),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return len(a_col)


def main() -> int:
    ap = argparse.ArgumentParser(description="C1 noise-schedule calibration vs real OCR.")
    ap.add_argument("--ocr", type=Path, default=Path("/tmp/c1cal/ocr_sample.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("/tmp/c1cal/calibration_report.json"))
    ap.add_argument("--corpus-out", type=Path, default=Path("/tmp/c1cal/pairs_calibrated.parquet"))
    ap.add_argument("--clean-from", type=Path, default=Path("/tmp/c1/pairs.parquet"))
    ap.add_argument("--max-anchors", type=int, default=4000)
    ap.add_argument("--k-per-anchor", type=int, default=len(list(NoiseMode)))
    ap.add_argument("--hard-negatives", type=int, default=7)
    ap.add_argument("--temp", type=float, default=0.15, help="softmax temperature for weights")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ocr_texts = _load_ocr_texts(args.ocr)
    if not ocr_texts:
        print(f"ERROR: no OCR text in {args.ocr}")
        return 3
    real = _agg([_stats(t) for t in ocr_texts])
    print(f"[cal] real-OCR strings={len(ocr_texts)}")

    anchors = _clean_anchors(args.clean_from, args.max_anchors)
    if not anchors:
        print("ERROR: no clean anchors available")
        return 3
    clean = _agg([_stats(a) for a in anchors])
    print(f"[cal] clean anchors={len(anchors)}")

    mode_sigs = _mode_signatures(anchors, args.seed)

    # Per-stat scale = max spread across {clean, real, all modes} so each stat
    # contributes comparably to the cosine score (avoids mean_token_len swamping).
    scales: dict[str, float] = {}
    for k in STAT_KEYS:
        vals = [clean[k], real[k], *(mode_sigs[m][k] for m in mode_sigs)]
        spread = max(vals) - min(vals)
        scales[k] = spread if spread > 1e-9 else 1.0

    real_delta = _normalize_deltas(real, clean, scales)
    mode_scores: dict[str, float] = {}
    mode_deltas: dict[str, dict[str, float]] = {}
    for m, sig in mode_sigs.items():
        d = _normalize_deltas(sig, clean, scales)
        mode_deltas[m] = d
        mode_scores[m] = _cosine(d, real_delta)

    weights = _softmax(mode_scores, args.temp)

    n_pairs = _write_corpus(
        anchors, weights, args.corpus_out, args.k_per_anchor, args.hard_negatives, args.seed
    )

    report = {
        "n_ocr_strings": len(ocr_texts),
        "n_clean_anchors": len(anchors),
        "real_ocr_stats": real,
        "clean_anchor_stats": clean,
        "mode_signatures": mode_sigs,
        "real_vs_clean_delta": real_delta,
        "mode_similarity_to_real": mode_scores,
        "calibrated_mode_weights": weights,
        "uniform_weight": 1.0 / len(mode_sigs),
        "softmax_temp": args.temp,
        "corpus_out": str(args.corpus_out),
        "corpus_pairs": n_pairs,
        "note": (
            "Weights nudge the v3 schedule toward real OCR by up-weighting the "
            "NoiseModes whose corruption signature best matches the measured "
            "real-OCR surface statistics. Committed diacritic_noise.py is "
            "unchanged; the nudge is applied only in this generated corpus."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[cal] wrote {args.report}")
    print(f"[cal] wrote nudged corpus {args.corpus_out} ({n_pairs} pairs)")
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    print("[cal] calibrated mode weights (uniform = %.3f):" % (1.0 / len(mode_sigs)))
    for m, w in ranked:
        print(f"        {m:>16}  w={w:.3f}  sim={mode_scores[m]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
