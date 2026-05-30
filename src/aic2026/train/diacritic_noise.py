# Implements SPEC-0014 section 3 + AC1 (the controlled diacritic-noise function).
"""A controlled Vietnamese diacritic-noise function.

C1 (proposal 08 part 3) trains a retrieval head to survive the diacritic
corruption Vietnamese ASR (PhoWhisper) and OCR (PaddleOCR) introduce. The
training signal is *generated*: take any clean Vietnamese string and produce
noisy variants by composing four modes.

Vietnamese marks live in two layers, both handled here via Unicode NFD:
  - **base modifiers** (breve, circumflex, horn) that form ``ă â ê ô ơ ư`` and
    the distinct letter ``đ``;
  - **tone marks** (grave, acute, hook-above, tilde, dot-below).

Everything is pure and deterministic given ``(text, mode, rng)``; non-Vietnamese
characters pass through unchanged; arbitrary Unicode never raises.
"""

from __future__ import annotations

import random
import unicodedata
from enum import StrEnum

# The five Vietnamese tone combining marks (level tone carries no mark).
# grave, acute, tilde, hook-above, dot-below.
TONE_MARKS: tuple[str, ...] = ("\u0300", "\u0301", "\u0303", "\u0309", "\u0323")
_TONE_SET = frozenset(TONE_MARKS)

# Special base letters that do not decompose under NFD but still carry a
# "diacritic" semantically: the Vietnamese đ/Đ.
_STROKE_D = {"\u0111": "d", "\u0110": "D"}  # đ -> d, Đ -> D


class NoiseMode(StrEnum):
    """The four diacritic-noise modes (proposal 08 part 3.2 step 1)."""

    DROP_ALL = "drop_all"  # strip every diacritic + tone -> "tre em"
    RANDOM_DROP = "random_drop"  # drop each mark w.p. p ~ Beta(2, 5) per string
    TONE_SWAP = "tone_swap"  # swap a present tone for a different tone
    MIXED = "mixed"  # random_drop composed with tone_swap


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def drop_all_diacritics(text: str) -> str:
    """Strip every Vietnamese diacritic and tone (NFD -> drop combining; đ->d).

    Idempotent: ``drop_all_diacritics(drop_all_diacritics(x)) == drop_all_diacritics(x)``.
    """
    out: list[str] = []
    for ch in _nfd(text):
        if unicodedata.combining(ch):
            continue
        out.append(_STROKE_D.get(ch, ch))
    return _nfc("".join(out))


def random_drop(text: str, p: float, rng: random.Random) -> str:
    """Drop each diacritic mark (combining mark or đ-stroke) with probability ``p``."""
    out: list[str] = []
    for ch in _nfd(text):
        if unicodedata.combining(ch):
            if rng.random() < p:
                continue
            out.append(ch)
        elif ch in _STROKE_D and rng.random() < p:
            out.append(_STROKE_D[ch])
        else:
            out.append(ch)
    return _nfc("".join(out))


def tone_swap(text: str, p: float, rng: random.Random) -> str:
    """Swap each present tone mark for a *different* tone with probability ``p``."""
    out: list[str] = []
    for ch in _nfd(text):
        if ch in _TONE_SET and rng.random() < p:
            alternatives = [t for t in TONE_MARKS if t != ch]
            out.append(rng.choice(alternatives))
        else:
            out.append(ch)
    return _nfc("".join(out))


def noise(text: str, mode: NoiseMode, *, rng: random.Random) -> str:
    """Apply one noise ``mode`` to ``text``.

    Deterministic given ``(text, mode, rng-state)``. For the probabilistic modes
    the per-string drop/swap rate is drawn from ``Beta(2, 5)`` (proposal 08
    part 3.2): mass concentrated on light corruption, with a tail of heavy noise.
    """
    if mode is NoiseMode.DROP_ALL:
        return drop_all_diacritics(text)
    if mode is NoiseMode.RANDOM_DROP:
        return random_drop(text, rng.betavariate(2, 5), rng)
    if mode is NoiseMode.TONE_SWAP:
        return tone_swap(text, rng.betavariate(2, 5), rng)
    if mode is NoiseMode.MIXED:
        dropped = random_drop(text, rng.betavariate(2, 5), rng)
        return tone_swap(dropped, rng.betavariate(2, 5), rng)
    raise ValueError(f"unknown noise mode: {mode!r}")  # pragma: no cover


def _rng_for(text: str, seed: int) -> random.Random:
    """A deterministic RNG keyed on ``(seed, text)``.

    ``random.Random`` seeded with a str uses a sha512-based, process-independent
    derivation (version 2), so this is reproducible across runs/machines.
    """
    return random.Random(f"{seed}\x00{text}")


def variants(text: str, *, k: int = 4, seed: int) -> list[tuple[str, NoiseMode]]:
    """Return ``k`` ``(noisy, mode)`` pairs for ``text``, cycling the four modes.

    Deterministic per ``(text, seed)``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    rng = _rng_for(text, seed)
    modes = list(NoiseMode)
    return [(noise(text, modes[i % len(modes)], rng=rng), modes[i % len(modes)]) for i in range(k)]
