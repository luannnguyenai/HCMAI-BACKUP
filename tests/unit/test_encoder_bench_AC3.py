# Proves SPEC-0025 AC3: run_qualitative over DummyEmbedder encoders writes an
# HTML contact sheet with one section per query and one row per encoder of
# exactly top_k thumbnails, gold-agnostic (no GT). CPU-only, no torch.

from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.embedding.dummy import DummyEmbedder
from aic2026.eval.encoder_bench import frame_id_for, run_qualitative


def _fake_corpus(tmp_path: Path, n: int = 6) -> list[Path]:
    """n distinct fake jpg files under per-video subdirs (distinct bytes)."""
    paths: list[Path] = []
    for i in range(n):
        vid = tmp_path / f"L25_V{i // 3:03d}"
        vid.mkdir(parents=True, exist_ok=True)
        p = vid / f"{i % 3:03d}.jpg"
        p.write_bytes(f"frame-{i}-content".encode())
        paths.append(p)
    return paths


def test_run_qualitative_html_structure_AC3(tmp_path: Path) -> None:
    docs = _fake_corpus(tmp_path, n=6)
    encoders = {
        "dummyA": DummyEmbedder(dim=64, model_id="dummy-A"),
        "dummyB": DummyEmbedder(dim=64, model_id="dummy-B"),
    }
    queries = ["con cho o cho", "ha noi mua thu", "pho bo"]
    out_html = tmp_path / "report" / "bench_report.html"

    results = run_qualitative(encoders, queries, docs, top_k=3, out_html=out_html)

    # Return structure: per encoder, per query, top_k hits.
    assert set(results.keys()) == {"dummyA", "dummyB"}
    for per_query in results.values():
        assert len(per_query) == len(queries)
        for hits in per_query:
            assert len(hits) == 3  # top_k
            for frame_id, path, score in hits:
                assert isinstance(frame_id, str) and isinstance(path, str)
                assert isinstance(score, float)

    html_text = out_html.read_text(encoding="utf-8")
    assert out_html.exists()
    assert "Query 1" in html_text and "Query 2" in html_text and "Query 3" in html_text
    assert "dummyA" in html_text and "dummyB" in html_text
    # 2 encoders x 3 queries x 3 top_k = 18 thumbnails.
    assert html_text.count("<img ") == 18
    # A known frame id appears (composite <videoDir>_<stem>).
    assert frame_id_for(docs[0]) == "L25_V000_000"


def test_run_qualitative_rejects_empty_AC3(tmp_path: Path) -> None:
    docs = _fake_corpus(tmp_path, n=2)
    enc = {"d": DummyEmbedder(dim=32)}
    with pytest.raises(ValueError, match="no query_texts"):
        run_qualitative(enc, [], docs, out_html=tmp_path / "r.html")
    with pytest.raises(ValueError, match="no doc_paths"):
        run_qualitative(enc, ["q"], [], out_html=tmp_path / "r.html")


def test_run_qualitative_topk_capped_to_doc_count_AC3(tmp_path: Path) -> None:
    docs = _fake_corpus(tmp_path, n=2)
    enc = {"d": DummyEmbedder(dim=32)}
    results = run_qualitative(enc, ["q1"], docs, top_k=5, out_html=tmp_path / "r.html")
    # only 2 docs -> at most 2 hits even though top_k=5
    assert len(results["d"][0]) == 2
