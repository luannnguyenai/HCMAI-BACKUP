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
    text_fields: tuple[str, ...]
    split: str = "train"


# The "captions + ASR" starting mix (user choice). HF ids are best-known and
# verified on the box at runtime; the builder skips any that fail to load.
# `<uit-openviic>` is an explicit placeholder (SPEC-0014 Q3).
DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("ai-enthusiasm-community/KTVIC", ("caption", "segment_caption")),
    SourceSpec("uitnlp/UIT-OpenViIC", ("caption",)),  # Q3: confirm id on the box
    SourceSpec("AILAB-VNUHCM/vivos", ("sentence",)),
    SourceSpec("linhtran92/viet_bud500", ("transcription",)),
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


def _harvest_source(spec: SourceSpec, *, max_rows: int | None) -> list[str]:
    """Load one HF dataset (streaming) and pull the configured text fields.

    Raises on any failure; the caller decides whether to skip.
    """
    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, split=spec.split, streaming=True)
    out: list[str] = []
    for row in ds:
        for fld in spec.text_fields:
            val = row.get(fld) if isinstance(row, dict) else None
            if isinstance(val, str) and val.strip():
                out.append(val)
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
