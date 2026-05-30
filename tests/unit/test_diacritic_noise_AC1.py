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


def test_ascii_passes_through_unchanged_AC1() -> None:
    s = "dog at Ben Thanh market 2026"
    rng = random.Random(0)
    for mode in NoiseMode:
        assert noise(s, mode, rng=random.Random(0)) == s  # no marks to touch
    assert drop_all_diacritics(s) == s
    _ = rng  # silence unused


def test_noise_is_deterministic_per_seed_AC1() -> None:
    text = "con chó ở chợ Bến Thành"
    for mode in NoiseMode:
        a = noise(text, mode, rng=random.Random("fixed-seed"))
        b = noise(text, mode, rng=random.Random("fixed-seed"))
        assert a == b


def test_variants_count_and_determinism_AC1() -> None:
    text = "Hà Nội mùa thu"
    v1 = variants(text, k=4, seed=7)
    v2 = variants(text, k=4, seed=7)
    assert len(v1) == 4
    assert v1 == v2  # deterministic per (text, seed)
    assert [m for _, m in v1] == list(NoiseMode)  # cycles the four modes
    # A different seed should (very likely) change at least one probabilistic variant.
    v3 = variants(text, k=4, seed=8)
    assert v1 != v3


def test_noise_never_raises_on_arbitrary_unicode_AC1() -> None:
    rng = random.Random(123)
    samples = [
        "",
        " ",
        "🙂🔥 emoji 漢字 кириллица",
        "\u0301\u0303 lone combining marks",
        "".join(chr(rng.randrange(0x20, 0x2FFF)) for _ in range(200)),
        "ﬁ ligatures and \t\n control \x00 chars",
    ]
    for s in samples:
        for mode in NoiseMode:
            out = noise(s, mode, rng=random.Random(1))
            assert isinstance(out, str)
