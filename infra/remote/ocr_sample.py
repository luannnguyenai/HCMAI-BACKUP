#!/usr/bin/env python3
"""Sample N keyframes, run Vietnamese OCR, dump recognized text (SPEC-0014 Q2).

Feeds the C1 noise calibration: the recognized text over real AIC keyframes is
the *real OCR-output distribution* we compare our synthetic `mixed_ocr` against
(`bin/train c1-calibrate --ocr <out>`). We don't need the whole 121k-frame
corpus - a few hundred text-bearing frames give stable surface statistics
(fragmentation, digit rate, casing).

Engines (our pipeline plans PaddleOCR PP-OCRv5; EasyOCR is an easier-to-install
fallback that is equally representative for *surface-stat* calibration):
  * ``--engine paddle`` (default): PaddleOCR, lang ``vi``. Handles both the 3.x
    ``.predict`` and 2.x ``.ocr`` return shapes.
  * ``--engine easy``: EasyOCR ``Reader(['vi']).readtext(..., detail=0)``.

One output line per text-bearing frame = the space-joined recognized strings
for that frame (this is what would land in the OCR index lane). Frames with no
detected text are skipped. Robust per-image: an OCR failure on one frame is
logged and skipped, never aborts the run.

Usage (on the box):
    uv run --with paddleocr --with paddlepaddle python ocr_sample.py \
        --kf-root /tmp/aic2025/kf --n 1000 --out /tmp/aic2025/ocr_sample.txt
    # fallback if PaddleOCR install is painful (torch already in the C1 env):
    uv run --with easyocr python ocr_sample.py --engine easy --n 1000 ...
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path


def _sample_images(kf_root: Path, n: int, seed: int) -> list[Path]:
    imgs = sorted(kf_root.rglob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no .jpg under {kf_root}")
    rng = random.Random(seed)
    rng.shuffle(imgs)
    return imgs[:n]


def _paddle_texts(ocr: object, path: Path) -> list[str]:
    """Recognized strings from a PaddleOCR result, tolerant of 2.x/3.x shapes."""
    # 3.x: ocr.predict(...) -> list[dict] with key "rec_texts".
    predict = getattr(ocr, "predict", None)
    if callable(predict):
        out: list[str] = []
        for res in predict(str(path)):
            texts = res.get("rec_texts") if isinstance(res, dict) else None
            if texts:
                out.extend(str(t) for t in texts)
        if out:
            return out
    # 2.x: ocr.ocr(path) -> [[ [box, (text, conf)], ... ]]
    legacy = ocr.ocr(str(path))  # type: ignore[attr-defined]
    out2: list[str] = []
    for page in legacy or []:
        for line in page or []:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)) and line[1]:
                out2.append(str(line[1][0]))
    return out2


def _build_paddle(lang: str):
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(lang=lang, use_textline_orientation=False)
    except TypeError:  # older signature
        return PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)


def _build_easy(lang: str):
    import easyocr

    return easyocr.Reader([lang])


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample keyframes -> Vietnamese OCR text.")
    ap.add_argument("--kf-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", choices=("paddle", "easy"), default="paddle")
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sample = _sample_images(args.kf_root, args.n, args.seed)
    print(f"sampled {len(sample)} frames from {args.kf_root} (engine={args.engine})")

    if args.engine == "paddle":
        engine = _build_paddle(args.lang)
        extract = lambda p: _paddle_texts(engine, p)  # noqa: E731
    else:
        reader = _build_easy(args.lang)
        extract = lambda p: [str(s) for s in reader.readtext(str(p), detail=0)]  # noqa: E731

    n_text = 0
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fh:
        for i, path in enumerate(sample, 1):
            try:
                texts = extract(path)
            except Exception as exc:  # one bad frame shouldn't abort the sweep
                print(f"  [skip] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            line = " ".join(t.strip() for t in texts if t and t.strip())
            if line:
                fh.write(line + "\n")
                n_text += 1
            if i % 100 == 0:
                rate = i / max(1e-6, time.time() - t0)
                print(f"  {i}/{len(sample)} done, {n_text} with text ({rate:.1f} img/s)")

    dt = time.time() - t0
    print(f"OK wrote {n_text} text-bearing lines of {len(sample)} frames -> {args.out} ({dt:.0f}s)")


if __name__ == "__main__":
    main()
