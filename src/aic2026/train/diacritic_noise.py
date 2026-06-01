# Implements SPEC-0014 section 3 + AC1 (the controlled diacritic + OCR noise function).
"""A controlled Vietnamese diacritic + OCR-error noise function.

C1 (proposal 08 part 3) trains a retrieval head to survive the systematic noise
Vietnamese ASR (PhoWhisper) and OCR (PaddleOCR, VietOCR) introduce. The training
signal is *generated*: take any clean Vietnamese string and produce noisy
variants by composing the modes below.

Two categories of noise are modelled:

  * **Diacritic noise** (the original C1 v1 schedule): drop / partial-drop /
    tone-swap / their combination. Mostly ASR-driven (PhoWhisper output) and
    aggressive-cleanup OCR. Marks live in two layers under NFD:
      - base modifiers (breve, circumflex, horn) that form ``ă â ê ô ơ ư`` and
        the distinct letter ``đ``;
      - tone marks (grave, acute, hook-above, tilde, dot-below).

  * **OCR character noise** (v2 schedule): real Vietnamese OCR also produces
    artifacts that are *not* diacritic-corruption -- specifically:
      - **letter-spacing**: ``quả táo`` -> ``q u ả  t á o`` when the segmenter
        over-splits glyphs (common on stylised fonts / tight kerning);
      - **visual confusables**: digits/letters that share a glyph shape get
        swapped: ``5 <-> s/S``, ``1 <-> l/I``, ``0 <-> o/O``, ``6 <-> G/b``,
        ``8 <-> B``, ``2 <-> Z/z``, ``9 <-> g/q``, ``7 <-> T``, ``4 <-> A``,
        ``rn <-> m``, ``cl <-> d``. These bite hardest on proper nouns +
        addresses + numerals.

  * **v3 schedule** (Tier A from the Hoang & Aw 2012 taxonomy survey):
      - **word merging** (Class 3 under-segmentation): inverse of space_split.
        ``quả táo`` -> ``quảtáo`` when the OCR or OCR-cleaner loses a space.
      - **homophone swap** (Class 2 real-syllable substitution; the dominant
        PhoWhisper failure): vowel-anchored tone re-assignment that can ADD a
        tone to a level syllable (``ma`` -> ``má``) as well as remove or swap
        an existing tone (``chợ`` -> ``cho``). The crucial distinction vs
        ``tone_swap`` is that the latter is mark-anchored (won't trigger on
        syllables without a current tone mark); ``homophone_swap`` is
        vowel-anchored so the full tonal-homophone family is reachable.
      - **case noise** (Class 3): per-word random case transform. Common on
        scene text (signs, banners) where OCR captures unusual case patterns.

Everything is pure and deterministic given ``(text, mode, rng)``; non-Vietnamese
characters pass through unchanged; arbitrary Unicode never raises.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Callable
from enum import StrEnum

# The five Vietnamese tone combining marks (level tone carries no mark).
# grave, acute, tilde, hook-above, dot-below.
TONE_MARKS: tuple[str, ...] = ("\u0300", "\u0301", "\u0303", "\u0309", "\u0323")
_TONE_SET = frozenset(TONE_MARKS)

# Special base letters that do not decompose under NFD but still carry a
# "diacritic" semantically: the Vietnamese đ/Đ.
_STROKE_D = {"\u0111": "d", "\u0110": "D"}  # đ -> d, Đ -> D

# OCR visual confusables. Bidirectional shape-confusion pairs commonly produced
# by Vietnamese OCR (PaddleOCR / VietOCR). Keys are the source character, value
# is the list of plausible misreads -- the noise picks uniformly when triggered.
# Bidirectional: include both directions explicitly so e.g. "5" can become "S"
# and "S" can become "5". Multi-char pairs (rn<->m, cl<->d) live in a separate
# table because they need lookahead handling.
_CONFUSABLES: dict[str, tuple[str, ...]] = {
    # digits <-> letters
    "0": ("O", "o"),  "O": ("0",),  "o": ("0",),
    "1": ("l", "I"),  "l": ("1", "I"),  "I": ("1", "l"),
    "2": ("Z", "z"),  "Z": ("2",),  "z": ("2",),
    "3": ("E",),  "E": ("3",),
    "4": ("A", "h"),  "A": ("4",),
    "5": ("S", "s"),  "S": ("5",),  "s": ("5",),
    "6": ("G", "b"),  "G": ("6",),  "b": ("6",),
    "7": ("T",),  "T": ("7",),
    "8": ("B",),  "B": ("8",),
    "9": ("g", "q"),  "g": ("9", "q"),  "q": ("9", "g"),
    # latin-only frequent confusions
    "c": ("e",),  "e": ("c",),
    "n": ("h",),  "h": ("n",),
}  # fmt: skip

# Two-character confusions handled with explicit lookahead. (Each side maps to
# the other so e.g. "rn" can read as "m" and "m" as "rn".)
_CONFUSABLES_2 = (
    ("rn", "m"),
    ("m", "rn"),
    ("cl", "d"),
    ("d", "cl"),
)

# Vietnamese vowel set used by ``homophone_swap``. Includes the ASCII vowels
# plus the four NFC base-modified vowels (ă, â, ê, ô, ơ, ư) so we can re-tone
# even a syllable that already has a base modifier but no tone (``cô`` -> ``cồ``).
# ``y`` is a Vietnamese semivowel that can carry tone and is included.
_VIETNAMESE_VOWELS: frozenset[str] = frozenset(
    "aeiouyAEIOUY"
    "\u0103\u00e2\u00ea\u00f4\u01a1\u01b0"  # ă â ê ô ơ ư
    "\u0102\u00c2\u00ca\u00d4\u01a0\u01af"  # Ă Â Ê Ô Ơ Ư
)


class NoiseMode(StrEnum):
    """The diacritic + OCR-noise modes (proposal 08 part 3.2 step 1, extended v3)."""

    # Diacritic-focused (v1)
    DROP_ALL = "drop_all"  # strip every diacritic + tone -> "tre em"
    RANDOM_DROP = "random_drop"  # drop each mark w.p. p ~ Beta(2, 5) per string
    TONE_SWAP = "tone_swap"  # swap a present tone for a different tone (mark-anchored)
    MIXED = "mixed"  # random_drop composed with tone_swap
    # OCR-character noise (v2)
    SPACE_SPLIT = "space_split"  # "quả táo" -> "q u ả  t á o" (per-word, w.p. p)
    CHAR_CONFUSE = "char_confuse"  # visual shape confusion (5<->S, 1<->l, ...)
    MIXED_OCR = "mixed_ocr"  # random_drop + space_split + char_confuse
    # OCR/ASR additions (v3 Tier A; Hoang & Aw 2012 Class 2/3)
    WORD_MERGE = "word_merge"  # Class 3 under-segmentation: "quả táo" -> "quảtáo"
    HOMOPHONE_SWAP = "homophone_swap"  # Class 2 ASR: ma <-> má <-> mã <-> mả ... (vowel-anchored)
    CASE_NOISE = "case_noise"  # Class 3: per-word random lower/upper/title


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


def space_split(text: str, p: float, rng: random.Random) -> str:
    """OCR over-segmentation: each whitespace-delimited word, w.p. ``p``, gets
    its characters separated by single spaces. ``quả táo`` -> ``q u ả  t á o``.

    Operates word-wise so multi-character glyph clusters (composed via NFC) stay
    together as the user sees them. Empty / single-char words are passed through.
    """
    if not text:
        return text
    out_words: list[str] = []
    for word in text.split(" "):
        if len(word) >= 2 and rng.random() < p:
            out_words.append(" ".join(word))
        else:
            out_words.append(word)
    return " ".join(out_words)


def _try_two_char_confuse(
    text: str, i: int, p: float, rng: random.Random
) -> tuple[str, int] | None:
    """If a two-char confusable starts at ``text[i]`` and the dice say so, return
    ``(replacement, advance)``; else ``None``. Caller advances ``i`` by ``advance``.
    """
    if i + 1 >= len(text):
        return None
    two = text[i : i + 2]
    for src, dst in _CONFUSABLES_2:
        if two == src and rng.random() < p:
            return dst, len(src)
    return None


def char_confuse(text: str, p: float, rng: random.Random) -> str:
    """Visual-confusable swaps (single + two-char). Each candidate is replaced w.p. ``p``.

    Tries the two-char patterns first (``rn`` <-> ``m``, ``cl`` <-> ``d``) so a
    single-char swap on the first character doesn't break the pair.
    """
    if not text or p <= 0:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        two = _try_two_char_confuse(text, i, p, rng)
        if two is not None:
            replacement, advance = two
            out.append(replacement)
            i += advance
            continue
        ch = text[i]
        opts = _CONFUSABLES.get(ch)
        if opts and rng.random() < p:
            out.append(rng.choice(opts))
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def word_merge(text: str, p: float, rng: random.Random) -> str:
    """OCR under-segmentation: drop the space between consecutive words w.p. ``p``.

    Symmetric inverse of ``space_split``. ``"quả táo Hà Nội"`` -> ``"quảtáo Hà Nội"``
    or ``"quả táoHà Nội"`` depending on which gaps fire. Single-word and empty
    inputs pass through.

    Implementation: split on a single space (preserves any other whitespace as-is
    inside words), then for each adjacent pair, with prob ``p`` join them with
    no space; otherwise re-insert the space. This is Class 3 from Hoang & Aw 2012.
    """
    if not text:
        return text
    words = text.split(" ")
    if len(words) < 2:
        return text
    out = [words[0]]
    for w in words[1:]:
        sep = "" if rng.random() < p else " "
        out.append(sep + w)
    return "".join(out)


def _random_case(word: str, rng: random.Random) -> str:
    """Apply one of {lower, upper, title} uniformly, excluding the identity."""
    candidates = []
    for transform in (str.lower, str.upper, str.title):
        candidate = transform(word)
        if candidate != word:
            candidates.append(candidate)
    if not candidates:
        return word
    return rng.choice(candidates)


def case_noise(text: str, p: float, rng: random.Random) -> str:
    """Per-word random casing: each word, w.p. ``p``, gets lower/upper/title applied.

    ``"Hà Nội mùa thu"`` p=1.0 -> e.g. ``"HÀ Nội mùa THU"``. The transform is
    chosen uniformly from {lower, upper, title} excluding the identity (so a
    triggered word always actually changes when the input has at least one
    transformable character). Class 3 from Hoang & Aw 2012.
    """
    if not text or p <= 0:
        return text
    return " ".join(_random_case(w, rng) if rng.random() < p else w for w in text.split(" "))


def _retone_at_vowel(decomposed: list[str], i: int, rng: random.Random, out: list[str]) -> int:
    """Consume the combining-mark run after ``decomposed[i]`` (a vowel), strip any
    existing tone, append a new tone different from the current one (or none),
    and return the next index to process. ``out`` is appended in place.
    """
    n = len(decomposed)
    current_tone: str | None = None
    j = i + 1
    while j < n and unicodedata.combining(decomposed[j]):
        if decomposed[j] in _TONE_SET and current_tone is None:
            current_tone = decomposed[j]  # captured + dropped from output
        else:
            out.append(decomposed[j])  # keep horn/breve/circumflex/etc.
        j += 1
    choices = [t for t in (None, *TONE_MARKS) if t != current_tone]
    new_tone = rng.choice(choices)
    if new_tone is not None:
        out.append(new_tone)
    return j


def homophone_swap(text: str, p: float, rng: random.Random) -> str:
    """Vowel-anchored tone re-assignment.

    For each Vietnamese vowel (in NFD), with prob ``p`` strip any combining tone
    mark immediately following it and append a new tone chosen uniformly from
    ``{None, *TONE_MARKS}`` constrained to differ from the current tone. Then
    NFC-recompose. This is the Class 2 ASR real-syllable error from Hoang & Aw
    2012: ``ma`` can become ``má`` / ``mã`` / ``mả`` / ``mạ`` / ``mà`` (the full
    tonal-homophone family) and ``chợ`` can become ``chơ`` / ``chò`` / ``chỏ`` etc.

    Crucial distinction vs ``tone_swap`` (mark-anchored): ``tone_swap`` only fires
    on syllables that already have a tone mark, so ``ma`` -> ``má`` is impossible
    there. ``homophone_swap`` is vowel-anchored, so the whole homophone family is
    reachable in both directions.

    Simplification: this is per-vowel, not per-syllable-nucleus. Syllables with
    multiple vowels (e.g. ``chuyển``) can therefore end up with multiple tone
    marks (``chùyển``), which is mechanically what real PhoWhisper sometimes
    produces when it mis-positions a tone. A linguistically-correct per-nucleus
    implementation would need full Vietnamese tone-placement rules; deferred.
    """
    if not text or p <= 0:
        return text
    decomposed = list(_nfd(text))
    out: list[str] = []
    i = 0
    n = len(decomposed)
    while i < n:
        ch = decomposed[i]
        out.append(ch)
        if ch in _VIETNAMESE_VOWELS and rng.random() < p:
            i = _retone_at_vowel(decomposed, i, rng, out)
        else:
            i += 1
    return _nfc("".join(out))


def _mixed(text: str, rng: random.Random) -> str:
    """Diacritic-focused mixed: random_drop + tone_swap."""
    dropped = random_drop(text, rng.betavariate(2, 5), rng)
    return tone_swap(dropped, rng.betavariate(2, 5), rng)


def _mixed_ocr(text: str, rng: random.Random) -> str:
    """Realistic OCR-output noise: diacritic drops + visual confusion + over-segmentation.

    Apply in this order so the spacing artifact is computed on the diacritic-
    + confusable-corrupted text (otherwise NFC reflow could re-glue split chars).
    """
    out = random_drop(text, rng.betavariate(2, 5), rng)
    out = char_confuse(out, rng.betavariate(2, 5), rng)
    return space_split(out, rng.betavariate(2, 5), rng)


# Dispatch table: NoiseMode -> (text, rng) -> str. Keeps `noise()` flat.
_NOISE_DISPATCH: dict[NoiseMode, Callable[[str, random.Random], str]] = {
    NoiseMode.DROP_ALL: lambda t, _rng: drop_all_diacritics(t),
    NoiseMode.RANDOM_DROP: lambda t, rng: random_drop(t, rng.betavariate(2, 5), rng),
    NoiseMode.TONE_SWAP: lambda t, rng: tone_swap(t, rng.betavariate(2, 5), rng),
    NoiseMode.MIXED: _mixed,
    NoiseMode.SPACE_SPLIT: lambda t, rng: space_split(t, rng.betavariate(2, 5), rng),
    NoiseMode.CHAR_CONFUSE: lambda t, rng: char_confuse(t, rng.betavariate(2, 5), rng),
    NoiseMode.MIXED_OCR: _mixed_ocr,
    # v3 Tier A
    NoiseMode.WORD_MERGE: lambda t, rng: word_merge(t, rng.betavariate(2, 5), rng),
    NoiseMode.HOMOPHONE_SWAP: lambda t, rng: homophone_swap(t, rng.betavariate(2, 5), rng),
    NoiseMode.CASE_NOISE: lambda t, rng: case_noise(t, rng.betavariate(2, 5), rng),
}


def noise(text: str, mode: NoiseMode, *, rng: random.Random) -> str:
    """Apply one noise ``mode`` to ``text``.

    Deterministic given ``(text, mode, rng-state)``. For the probabilistic modes
    the per-string drop/swap rate is drawn from ``Beta(2, 5)`` (proposal 08
    part 3.2): mass concentrated on light corruption, with a tail of heavy noise.
    """
    fn = _NOISE_DISPATCH.get(mode)
    if fn is None:
        raise ValueError(f"unknown noise mode: {mode!r}")  # pragma: no cover
    return fn(text, rng)


def _rng_for(text: str, seed: int) -> random.Random:
    """A deterministic RNG keyed on ``(seed, text)``.

    ``random.Random`` seeded with a str uses a sha512-based, process-independent
    derivation (version 2), so this is reproducible across runs/machines.
    """
    return random.Random(f"{seed}\x00{text}")


def variants(text: str, *, k: int | None = None, seed: int) -> list[tuple[str, NoiseMode]]:
    """Return ``k`` ``(noisy, mode)`` pairs for ``text``, cycling through ``NoiseMode``.

    Default ``k = len(NoiseMode)`` so every mode is represented per anchor (the
    v2 schedule has 7 modes). Pass an explicit ``k`` to truncate or extend the
    cycle. Deterministic per ``(text, seed)``.
    """
    modes = list(NoiseMode)
    if k is None:
        k = len(modes)
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    rng = _rng_for(text, seed)
    return [(noise(text, modes[i % len(modes)], rng=rng), modes[i % len(modes)]) for i in range(k)]
