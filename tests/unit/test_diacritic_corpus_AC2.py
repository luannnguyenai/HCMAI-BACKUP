# Proves SPEC-0014 AC2: build_corpus turns clean strings into a Parquet of
# (anchor_clean, positive_noisy, mode, hard_negs) with k positives per anchor,
# deduped; and a source that fails to load is skipped (non-fatal). CPU-only,
# no network (clean_strings override + a monkeypatched harvester).

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from aic2026.train import diacritic_corpus as dc
from aic2026.train.diacritic_corpus import SourceSpec, build_corpus, read_pairs
from aic2026.train.diacritic_noise import NoiseMode


def test_build_corpus_from_clean_strings_AC2(tmp_path: Path) -> None:
    clean = [
        "con chó ở chợ Bến Thành",
        "Hà Nội mùa thu lá vàng",
        "con chó ở chợ Bến Thành",  # exact dup -> dropped
        "ab",  # too short -> dropped
        "Đà Nẵng biển xanh cát trắng",
    ]
    out = tmp_path / "pairs.parquet"
    res = build_corpus(out=out, k=4, hard_negatives=7, seed=0, clean_strings=clean)

    assert out.exists()
    assert res.n_clean == 3  # dup + short removed
    assert res.n_pairs == 3 * 4
    assert res.sources_used == ["<clean_strings>"]
    assert res.sources_skipped == []

    rows = read_pairs(out)
    assert len(rows) == 12
    assert set(rows[0].keys()) == {"anchor_clean", "positive_noisy", "mode", "hard_negs"}

    # Exactly 4 positives per anchor, cycling the four modes.
    from collections import Counter

    per_anchor = Counter(r["anchor_clean"] for r in rows)
    assert set(per_anchor.values()) == {4}
    assert {r["mode"] for r in rows} == {m.value for m in NoiseMode}

    # Hard negatives are other corpus strings, never the anchor itself.
    for r in rows:
        assert isinstance(r["hard_negs"], list)
        assert r["anchor_clean"] not in r["hard_negs"]


def test_build_corpus_skips_failing_source_AC2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_harvest(spec: SourceSpec, *, max_rows: int | None) -> list[str]:
        if spec.hf_id == "bad/repo":
            raise RuntimeError("dataset not found")
        return ["con mèo trên mái nhà", "trời mưa rất to ở Sài Gòn"]

    monkeypatch.setattr(dc, "_harvest_source", fake_harvest)

    out = tmp_path / "pairs.parquet"
    res = build_corpus(
        sources=[SourceSpec("bad/repo", ("x",)), SourceSpec("good/repo", ("y",))],
        out=out,
        k=4,
        seed=1,
    )
    assert res.sources_used == ["good/repo"]
    assert res.sources_skipped == ["bad/repo"]
    assert res.n_clean == 2
    assert res.n_pairs == 8
    assert out.exists()


def test_build_corpus_all_sources_skipped_raises_AC2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(spec: SourceSpec, *, max_rows: int | None) -> list[str]:
        raise RuntimeError("nope")

    monkeypatch.setattr(dc, "_harvest_source", boom)
    with pytest.raises(ValueError, match="no clean strings"):
        build_corpus(sources=[SourceSpec("a/b", ("x",))], out=tmp_path / "x.parquet")
