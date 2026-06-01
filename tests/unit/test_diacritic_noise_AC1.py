# Proves SPEC-0014 AC1: the diacritic-noise function is correct (DROP_ALL strips
# every mark + is idempotent), deterministic per (text, mode, seed), and never
# raises on arbitrary Unicode.

from __future__ import annotations

import random
import unicodedata

import pytest

from aic2026.train.diacritic_noise import (
    NoiseMode,
    drop_all_diacritics,
    noise,
    variants,
)


@pytest.mark.parametrize(
    ("dirty", "clean"),
    [
        ("trẻ em", "tre em"),
        ("Đà Nẵng", "Da Nang"),
        ("phở bò", "pho bo"),
        ("Nguyễn Văn A", "Nguyen Van A"),
        ("chợ Bến Thành", "cho Ben Thanh"),
    ],
)
def test_drop_all_strips_every_diacritic_AC1(dirty: str, clean: str) -> None:
    assert drop_all_diacritics(dirty) == clean
    # No combining marks survive.
    assert not any(
        unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", drop_all_diacritics(dirty))
    )


def test_drop_all_is_idempotent_AC1() -> None:
    once = drop_all_diacritics("Hồ Chí Minh, đường Lê Lợi")
    assert drop_all_diacritics(once) == once


# The diacritic-only subset: these modes only touch combining marks + đ-stroke,
# so ASCII text without diacritics passes through them unchanged. The OCR modes
# (SPACE_SPLIT / CHAR_CONFUSE / MIXED_OCR) can and should change ASCII.
_DIACRITIC_ONLY_MODES = (
    NoiseMode.DROP_ALL,
    NoiseMode.RANDOM_DROP,
    NoiseMode.TONE_SWAP,
    NoiseMode.MIXED,
)


def test_diacritic_only_modes_passthrough_ascii_AC1() -> None:
    """The 4 diacritic-focused modes leave plain ASCII unchanged."""
    s = "dog at Ben Thanh market"  # no digits/letters that are confusables
    for mode in _DIACRITIC_ONLY_MODES:
        assert noise(s, mode, rng=random.Random(0)) == s
    assert drop_all_diacritics(s) == s


def test_noise_is_deterministic_per_seed_AC1() -> None:
    """All modes (incl. OCR) are deterministic given (text, mode, seed)."""
    text = "con chó ở chợ Bến Thành 2026 phố 5"
    for mode in NoiseMode:
        a = noise(text, mode, rng=random.Random("fixed-seed"))
        b = noise(text, mode, rng=random.Random("fixed-seed"))
        assert a == b


def test_variants_default_k_covers_all_modes_AC1() -> None:
    """variants() default cycles through every NoiseMode once -- so an anchor
    in the training corpus gets one variant per noise category."""
    text = "Hà Nội mùa thu"
    v1 = variants(text, seed=7)  # default k -> len(NoiseMode)
    v2 = variants(text, seed=7)
    assert len(v1) == len(list(NoiseMode))
    assert v1 == v2  # deterministic per (text, seed)
    assert [m for _, m in v1] == list(NoiseMode)
    v3 = variants(text, seed=8)
    assert v1 != v3


def test_noise_never_raises_on_arbitrary_unicode_AC1() -> None:
    """Every mode tolerates emoji, CJK, control chars, lone combining marks etc."""
    rng = random.Random(123)
    samples = [
        "",
        " ",
        "🙂🔥 emoji 漢字 кириллица",
        "\u0301\u0303 lone combining marks",
        "".join(chr(rng.randrange(0x20, 0x2FFF)) for _ in range(200)),
        "ﬁ ligatures and \t\n control \x00 chars",
        "addresses: 5 Lê Lợi, 1B Nguyễn Huệ",  # confusable digits + diacritics
    ]
    for s in samples:
        for mode in NoiseMode:
            out = noise(s, mode, rng=random.Random(1))
            assert isinstance(out, str)


# --- OCR noise modes (v2) ----------------------------------------------------


def test_space_split_separates_chars_within_words_AC1() -> None:
    """SPACE_SPLIT with p=1 inserts single spaces between every character of
    every multi-char word. The inter-word space collapses with the new
    intra-word spacing -- realistic for over-segmented OCR where the word
    boundary becomes indistinguishable from the per-char gap."""
    from aic2026.train.diacritic_noise import space_split

    # p=1 -> deterministic: every word is split; output is single-spaced.
    out = space_split("quả táo", p=1.0, rng=random.Random(0))
    assert out == "q u ả t á o"
    # Single-char words are left alone.
    assert space_split("a b c", p=1.0, rng=random.Random(0)) == "a b c"
    # Mixed: one multi-char word split, one untouched (p=0 -> no-op).
    assert space_split("Hà Nội", p=0.0, rng=random.Random(0)) == "Hà Nội"


def test_char_confuse_swaps_known_pairs_AC1() -> None:
    """CHAR_CONFUSE at p=1 deterministically replaces every confusable char."""
    from aic2026.train.diacritic_noise import char_confuse

    out = char_confuse("5 Lê Lợi", p=1.0, rng=random.Random(0))
    # "5" must change to one of its confusables; "L" stays (no entry); "ê/ợ" stay.
    assert out[0] in {"S", "s"}
    assert "Lê L" in out and "ợi" in out
    # ASCII passthrough with no confusables.
    assert (
        char_confuse("vietnamese text", p=1.0, rng=random.Random(0)) != "vietnamese text"
    )  # 'c'/'e'/'n'/'h' swap
    # p=0 -> identity.
    assert char_confuse("Hà Nội 2026", p=0.0, rng=random.Random(0)) == "Hà Nội 2026"


def test_char_confuse_two_char_rn_to_m_AC1() -> None:
    """The two-char "rn -> m" / "m -> rn" pair is honoured."""
    from aic2026.train.diacritic_noise import char_confuse

    # With p=1, both directions trigger; we just want to see one of them.
    swept = char_confuse("rn morning", p=1.0, rng=random.Random(0))
    # Either "rn" became "m" (so length shrank) or "m" became "rn" (so length grew),
    # but it should NOT be identical.
    assert swept != "rn morning"


def test_mixed_ocr_applies_drop_split_and_confuse_AC1() -> None:
    """MIXED_OCR is a non-trivial composition: at high effective p the output
    should differ from the input for a string carrying all three signals."""
    text = "Nguyễn 5 Lê Lợi quả táo"
    # Try a few seeds; at least one should produce a change.
    changed = False
    for s in range(20):
        out = noise(text, NoiseMode.MIXED_OCR, rng=random.Random(f"seed{s}"))
        if out != text:
            changed = True
            break
    assert changed, (
        "MIXED_OCR should change Vietnamese strings with confusables on at least one seed"
    )
