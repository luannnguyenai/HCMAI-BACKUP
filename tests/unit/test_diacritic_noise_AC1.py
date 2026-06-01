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


# --- v3 Tier A modes (WORD_MERGE / HOMOPHONE_SWAP / CASE_NOISE) -------------


def test_word_merge_drops_spaces_with_p1_AC1() -> None:
    """WORD_MERGE with p=1 removes every inter-word space; p=0 is identity."""
    from aic2026.train.diacritic_noise import word_merge

    # p=1 -> every gap collapses; the words concatenate with no spaces.
    assert word_merge("a b c", p=1.0, rng=random.Random(0)) == "abc"
    assert word_merge("quả táo Hà Nội", p=1.0, rng=random.Random(0)) == "quảtáoHàNội"
    # p=0 -> identity.
    assert word_merge("quả táo Hà Nội", p=0.0, rng=random.Random(0)) == "quả táo Hà Nội"
    # Empty / single-word inputs pass through.
    assert word_merge("", p=1.0, rng=random.Random(0)) == ""
    assert word_merge("solo", p=1.0, rng=random.Random(0)) == "solo"


def test_case_noise_changes_case_AC1() -> None:
    """CASE_NOISE at p=1 changes case on every word (excluding identity transforms)."""
    from aic2026.train.diacritic_noise import case_noise

    # p=1 + a seeded RNG: at least one word must change.
    text = "Hà Nội mùa thu lá vàng"
    out = case_noise(text, p=1.0, rng=random.Random(0))
    assert out != text
    # ASCII works too.
    assert case_noise("hello world", p=1.0, rng=random.Random(0)) != "hello world"
    # p=0 -> identity.
    assert case_noise(text, p=0.0, rng=random.Random(0)) == text


def test_homophone_swap_can_add_tone_to_level_syllable_AC1() -> None:
    """HOMOPHONE_SWAP can add a tone to a level (untoned) syllable.

    This is the CORE distinction from ``tone_swap``: tone_swap is mark-anchored
    (only fires on existing tone marks), so ``ma`` (no tone) cannot become
    ``má``/``mã``/etc. through it. HOMOPHONE_SWAP is vowel-anchored and reaches
    the full homophone family. The test asserts that across multiple seeds, at
    least one toned variant of ``ma`` is produced.
    """
    from aic2026.train.diacritic_noise import homophone_swap

    toned_variants = {"\u00e0", "\u00e1", "\u00e3", "\u1ea3", "\u1ea1"}  # à á ã ả ạ
    saw_toned = False
    for s in range(50):
        out = homophone_swap("ma", p=1.0, rng=random.Random(f"seed{s}"))
        # Output is exactly one syllable; check if char[1] is a toned variant of 'a'.
        if any(t in out for t in toned_variants):
            saw_toned = True
            break
    assert saw_toned, "homophone_swap on 'ma' must occasionally produce a toned variant"


def test_homophone_swap_can_remove_existing_tone_AC1() -> None:
    """HOMOPHONE_SWAP can also strip an existing tone (chợ -> chơ across seeds).

    Note: horn / breve / circumflex are BASE-vowel modifiers (part of the vowel
    identity ơ vs o), not tones. homophone_swap preserves them and only re-tones,
    so ``chợ`` (cơ + dot-below) -> ``chơ`` (cơ + no tone) when the chosen new
    tone is None. ``chợ`` -> ``cho`` would require also stripping the horn,
    which is a vowel-identity change (covered by ``drop_all`` / ``random_drop``)
    rather than a tone change. The semantics here are tone-only.
    """
    from aic2026.train.diacritic_noise import homophone_swap

    saw_untoned = False
    for s in range(50):
        out = homophone_swap("chợ", p=1.0, rng=random.Random(f"seed{s}"))
        if out == "chơ":
            saw_untoned = True
            break
    assert saw_untoned, "homophone_swap on 'chợ' must occasionally strip the tone -> 'chơ'"


def test_homophone_swap_passes_through_consonants_and_emoji_AC1() -> None:
    """Non-vowel characters and emoji/CJK pass through untouched."""
    from aic2026.train.diacritic_noise import homophone_swap

    # All-consonant string: no vowels -> identity even at p=1.
    assert homophone_swap("bcdfgh", p=1.0, rng=random.Random(0)) == "bcdfgh"
    # Emoji + CJK: vowels in there might mutate, but emoji/CJK must survive.
    out = homophone_swap("🙂🔥 漢字", p=1.0, rng=random.Random(0))
    assert "🙂" in out and "🔥" in out and "漢" in out and "字" in out
    # p=0 -> identity for any input.
    assert homophone_swap("chợ Hà Nội ma", p=0.0, rng=random.Random(0)) == "chợ Hà Nội ma"
