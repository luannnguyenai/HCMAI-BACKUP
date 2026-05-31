# Implements SPEC-0014 section 3 + AC2 (contrastive-corpus builder).
"""Build the C1 contrastive corpus: clean Vietnamese text -> noisy variants.

Clean strings come from a configurable set of public HuggingFace datasets
(default: KTVIC + UIT-OpenViIC captions, VIVOS + Bud500 ASR transcripts - the
"captions + ASR" mix). Pre-June-25 this is the *only* data C1 needs; post-June-25
the same builder runs over our own index text via the ``clean_strings`` override.

The builder is **fault-tolerant per source** (a dataset that fails to load is
skipped, not fatal - same contract as the ``cache-weights`` job) and writes a
Parquet of ``(anchor_clean, positive_noisy, mode, hard_negs)`` rows.

`datasets` and `pyarrow` are only imported where used so importing this module
stays light (the registry import path must not require the ``train`` extra).
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aic2026.train.diacritic_noise import variants

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceSpec:
    """One public HF dataset to harvest clean Vietnamese strings from."""

    hf_id: str
    text_fields: tuple[str, ...] = ()  # empty -> auto-detect prose columns
    split: str = "train"
    config: str | None = None  # HF dataset config (e.g. "20231101.vi")
    split_sentences: bool = False  # split long-text fields into sentence chunks


# Verified on the H200 lease (SPEC-0014 Q3): the first smoke showed KTVIC has no
# flat `caption` field (it's an image-caption set), UIT-OpenViIC's id was wrong,
# VIVOS ships a loader *script* (datasets >= 4.x dropped script support), and
# Bud500 is gated. So the reliable anchor is Vietnamese Wikipedia (ungated,
# parquet-native, full-diacritic prose); KTVIC stays best-effort via column
# auto-detection. Gated ASR sets (Bud500, common_voice) are opt-in once the box
# has an HF token with access. The builder skips any source that fails to load.
DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "wikimedia/wikipedia",
        text_fields=("text",),
        config="20231101.vi",
        split_sentences=True,
    ),
    SourceSpec("ai-enthusiasm-community/KTVIC"),  # auto-detect the caption column
)


@dataclass
class CorpusResult:
    """Outcome of a corpus build."""

    n_clean: int
    n_pairs: int
    out: Path
    sources_used: list[str] = field(default_factory=list)
    sources_skipped: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _dedup(strings: Iterable[str], *, min_chars: int = 4) -> list[str]:
    """Normalise, drop short/empty, dedup case-insensitively (keep first)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in strings:
        if not isinstance(raw, str):
            continue
        norm = _normalize(raw)
        if len(norm) < min_chars:
            continue
        key = norm.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


_SENT_SPLIT = re.compile(r"[.!?\n]+")


def _split_into_sentences(text: str, *, min_chars: int = 15, max_chars: int = 300) -> list[str]:
    """Split long prose (e.g. a Wikipedia article) into sentence-ish chunks."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if min_chars <= len(s.strip()) <= max_chars]


def _looks_like_prose(value: str) -> bool:
    """Heuristic for auto-detecting caption/sentence columns (vs ids/urls/paths)."""
    v = value.strip()
    return " " in v and len(v) >= 15 and "://" not in v


def _row_strings(row: dict, spec: SourceSpec) -> list[str]:
    """Pull candidate strings from one row, honouring text_fields or auto-detecting."""
    raw: list[str] = []
    if spec.text_fields:
        for fld in spec.text_fields:
            val = row.get(fld)
            if isinstance(val, str):
                raw.append(val)
            elif isinstance(val, list):
                raw.extend(x for x in val if isinstance(x, str))
    else:  # auto-detect: any string-ish column that reads like prose
        for val in row.values():
            if isinstance(val, str) and _looks_like_prose(val):
                raw.append(val)
            elif isinstance(val, list):
                raw.extend(x for x in val if isinstance(x, str) and _looks_like_prose(x))
    if spec.split_sentences:
        out: list[str] = []
        for s in raw:
            out.extend(_split_into_sentences(s))
        return out
    return [s for s in raw if s.strip()]


def _harvest_source(spec: SourceSpec, *, max_rows: int | None) -> list[str]:
    """Load one HF dataset (streaming) and pull text. Raises on failure (caller skips)."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or None
    ds = load_dataset(spec.hf_id, spec.config, split=spec.split, streaming=True, token=token)

    # Turn off image decoding on image-caption sets (KTVIC): we only want the
    # text columns, and decoding every image is slow and needs Pillow.
    feats = getattr(ds, "features", None)
    if feats:
        from datasets import Image as HfImage

        for fname, feat in list(feats.items()):
            if feat.__class__.__name__ == "Image":
                ds = ds.cast_column(fname, HfImage(decode=False))

    out: list[str] = []
    for row in ds:
        if isinstance(row, dict):
            out.extend(_row_strings(row, spec))
        if max_rows is not None and len(out) >= max_rows:
            break
    return out


def _collect_clean(
    sources: Sequence[SourceSpec],
    *,
    max_per_source: int | None,
    used: list[str],
    skipped: list[str],
) -> list[str]:
    collected: list[str] = []
    for spec in sources:
        try:
            rows = _harvest_source(spec, max_rows=max_per_source)
        except Exception as exc:  # any load failure (incl. missing `datasets`) is non-fatal
            logger.warning("skipping source %s: %s", spec.hf_id, exc)
            skipped.append(spec.hf_id)
            continue
        if not rows:
            logger.warning("source %s yielded no rows; skipping", spec.hf_id)
            skipped.append(spec.hf_id)
            continue
        used.append(spec.hf_id)
        collected.extend(rows)
    return collected


def _mine_hard_negatives(clean: Sequence[str], *, n: int, rng: random.Random) -> list[list[str]]:
    """Random in-corpus negatives per anchor (v1).

    SPEC-0014 Q4: the production version mines top-N BGE-M3 neighbours of the
    clean anchor; that needs a model forward pass and is a follow-up. Random
    in-corpus negatives are the documented CPU/CI fallback and a fine starting
    signal alongside in-batch negatives during training.
    """
    if len(clean) <= 1 or n <= 0:
        return [[] for _ in clean]
    out: list[list[str]] = []
    for i in range(len(clean)):
        negs: list[str] = []
        guard = 0
        while len(negs) < min(n, len(clean) - 1) and guard < n * 20:
            guard += 1
            j = rng.randrange(len(clean))
            if j != i and clean[j] not in negs:
                negs.append(clean[j])
        out.append(negs)
    return out


def build_corpus(
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
    *,
    out: Path,
    k: int = 4,
    max_per_source: int | None = None,
    hard_negatives: int = 7,
    seed: int = 0,
    clean_strings: Sequence[str] | None = None,
) -> CorpusResult:
    """Harvest clean strings -> dedup -> k noisy variants -> mine negatives -> Parquet.

    ``clean_strings`` bypasses HF entirely (used by CI fixtures and, post-June-25,
    by the path that feeds our own index text in). Any source that errors on load
    is recorded in ``sources_skipped`` and the build continues.

    Parquet columns: ``anchor_clean`` (str), ``positive_noisy`` (str),
    ``mode`` (str), ``hard_negs`` (list[str]).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    used: list[str] = []
    skipped: list[str] = []

    if clean_strings is not None:
        raw_clean: list[str] = list(clean_strings)
        used.append("<clean_strings>")
    else:
        raw_clean = _collect_clean(
            sources, max_per_source=max_per_source, used=used, skipped=skipped
        )

    clean = _dedup(raw_clean)
    if not clean:
        raise ValueError(
            f"no clean strings collected (all sources skipped or empty). skipped={skipped}"
        )

    rng = random.Random(f"{seed}\x00hardnegs")
    hard = _mine_hard_negatives(clean, n=hard_negatives, rng=rng)

    anchors: list[str] = []
    positives: list[str] = []
    modes: list[str] = []
    negs_col: list[list[str]] = []
    for anchor, negs in zip(clean, hard, strict=True):
        for noisy, mode in variants(anchor, k=k, seed=seed):
            anchors.append(anchor)
            positives.append(noisy)
            modes.append(mode.value)
            negs_col.append(negs)

    table = pa.table(
        {
            "anchor_clean": pa.array(anchors, pa.string()),
            "positive_noisy": pa.array(positives, pa.string()),
            "mode": pa.array(modes, pa.string()),
            "hard_negs": pa.array(negs_col, pa.list_(pa.string())),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)

    logger.info(
        "built corpus: %d clean -> %d pairs -> %s (used=%s skipped=%s)",
        len(clean),
        len(anchors),
        out,
        used,
        skipped,
    )
    return CorpusResult(
        n_clean=len(clean),
        n_pairs=len(anchors),
        out=out,
        sources_used=used,
        sources_skipped=skipped,
    )


def read_pairs(path: Path) -> list[dict[str, object]]:
    """Read a corpus Parquet back into a list of row dicts (training + tests)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()
