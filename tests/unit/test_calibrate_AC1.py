# Proves SPEC-0014 Q2 calibration harness: profile_text computes sane surface
# statistics, single-char-token ratio detects OCR over-segmentation, synthetic
# noise profiling runs over all modes, and compare() raises the right advisory
# flags for anchor/query length mismatch and mixed_ocr over-fragmentation.
# Pure stdlib + the noise module - no torch / pyarrow.

from __future__ import annotations

import pytest

from aic2026.train.calibrate import (
    TextStats,
    compare,
    load_strings,
    profile_synthetic_noise,
    profile_text,
)
from aic2026.train.diacritic_noise import NoiseMode

_VI = [
    "Hà Nội là thủ đô của Việt Nam",
    "Người phụ nữ mặc áo dài trắng đi trên cầu Long Biên",
    "Số 5 đường Lê Lợi, Quận 1",
]


def test_profile_text_basic_AC1() -> None:
    st = profile_text(_VI)
    assert st.n == 3
    assert st.char_len_mean > 0
    assert 0.0 <= st.diacritic_ratio_mean <= 1.0
    # These Vietnamese strings are diacritic-dense.
    assert st.diacritic_ratio_mean > 0.2
    # No spaces inserted mid-word -> very low single-char-token ratio.
    assert st.single_char_token_ratio_mean < 0.1


def test_profile_text_rejects_empty_AC1() -> None:
    with pytest.raises(ValueError, match="no non-empty"):
        profile_text(["", "   ", "\n"])


def test_single_char_token_ratio_detects_fragmentation_AC1() -> None:
    clean = profile_text(["quả táo đỏ trên bàn"])
    fragmented = profile_text(["q u ả t á o đ ỏ t r ê n b à n"])
    # Over-segmented OCR-style text has a far higher single-char-token ratio.
    assert fragmented.single_char_token_ratio_mean > clean.single_char_token_ratio_mean
    assert fragmented.single_char_token_ratio_mean > 0.8


def test_profile_synthetic_noise_covers_all_modes_AC1() -> None:
    syn = profile_synthetic_noise(_VI, seed=0, max_anchors=None)
    assert set(syn.keys()) == {m.value for m in NoiseMode}
    for st in syn.values():
        assert isinstance(st, TextStats)
        assert st.n == len(_VI)
    # space_split / mixed_ocr should fragment more than the diacritic-only drop_all.
    assert (
        syn["space_split"].single_char_token_ratio_mean
        >= syn["drop_all"].single_char_token_ratio_mean
    )


def test_profile_synthetic_noise_is_deterministic_AC1() -> None:
    a = profile_synthetic_noise(_VI, seed=7)
    b = profile_synthetic_noise(_VI, seed=7)
    assert a == b


def test_compare_flags_anchor_length_mismatch_AC1() -> None:
    # Real queries ~10 chars; our anchors ~100 chars -> >2x length flag.
    short_q = profile_text(["áo đỏ", "xe máy", "nhà cao"])
    long_anchor = profile_text(["w" * 100, "x" * 110, "y" * 95])
    rep = compare(real_query=short_q, our_anchor=long_anchor, real_ocr=None, synthetic=None)
    assert any("anchor length mismatch" in f for f in rep["flags"])
    assert rep["verdict"].startswith("1") or "flag" in rep["verdict"]


def test_compare_flags_mixed_ocr_fragmentation_AC1() -> None:
    # Synthetic mixed_ocr heavily fragmented; real OCR barely -> over-fragment flag.
    syn = {"mixed_ocr": profile_text(["q u ả t á o đ ỏ b à n c a o"])}
    real_ocr = profile_text(["quả táo đỏ trên bàn cao"])
    rep = compare(real_query=None, our_anchor=None, real_ocr=real_ocr, synthetic=syn)
    assert any("over-fragments" in f for f in rep["flags"])


def test_compare_clean_when_aligned_AC1() -> None:
    st = profile_text(_VI)
    # Same distribution on both sides -> no flags.
    rep = compare(real_query=st, our_anchor=st, real_ocr=None, synthetic=None)
    assert rep["flags"] == []
    assert rep["verdict"] == "calibration OK (no flags)"


def test_load_strings_txt_and_json_AC1(tmp_path) -> None:
    txt = tmp_path / "q.txt"
    txt.write_text("dòng một\ndòng hai\n\n", encoding="utf-8")
    assert load_strings(txt) == ["dòng một", "dòng hai"]

    js = tmp_path / "q.json"
    js.write_text('{"queries": ["truy vấn một", "x", "truy vấn hai"]}', encoding="utf-8")
    loaded = load_strings(js)
    # "x" is too short (<4 chars) and is dropped.
    assert "truy vấn một" in loaded
    assert "truy vấn hai" in loaded
    assert "x" not in loaded

    # Directory walk picks up both.
    assert len(load_strings(tmp_path)) >= 4


def test_load_strings_xlsx_AC1(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")

    xlsx = tmp_path / "DanhSachTruyVanAIC_Chungket.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "Người đàn ông mặc áo xanh đi xe đạp qua cầu Long Biên"
    ws["A2"] = "Cô gái cầm ô vàng đứng trước cửa hàng tạp hóa"
    ws["A3"] = "abc"  # short string (<4 chars) -> dropped by the length filter
    ws["A4"] = 12345  # numeric cell -> skipped (would be "12345" if not skipped)
    wb.save(xlsx)

    loaded = load_strings(xlsx)
    assert "Người đàn ông mặc áo xanh đi xe đạp qua cầu Long Biên" in loaded
    assert "Cô gái cầm ô vàng đứng trước cửa hàng tạp hóa" in loaded
    # Short string and numeric cell are filtered out.
    assert "abc" not in loaded
    assert "12345" not in loaded
    assert all(isinstance(s, str) for s in loaded)

    # A directory containing the spreadsheet is walked correctly.
    dir_loaded = load_strings(tmp_path)
    assert "Người đàn ông mặc áo xanh đi xe đạp qua cầu Long Biên" in dir_loaded
    assert "Cô gái cầm ô vàng đứng trước cửa hàng tạp hóa" in dir_loaded
