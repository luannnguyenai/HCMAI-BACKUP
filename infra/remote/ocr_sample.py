#!/usr/bin/env python
# Implements SPEC-0014 calibration support (C1 noise-schedule calibration).
# Uncommitted infra helper (NOT part of the shipped package): sample N keyframes
# from the AIC2025 proxy keyframe tree, run OCR over them on a pinned GPU, and
# write a JSONL of the recognised Vietnamese surface text. This real-OCR output
# is the empirical target that c1_calibrate.py compares the synthetic C1 noise
# schedule against.
#
#   CUDA_VISIBLE_DEVICES=7 uv run python infra/remote/ocr_sample.py \
#       --kf-root /tmp/aic2025/kf --n 400 --out /tmp/c1cal/ocr_sample.jsonl
#
# OCR backend: PaddleOCR had a CPU-only bug on this box before, so EasyOCR is the
# default fallback (GPU when a CUDA device is visible). Pass --backend paddle to
# try PaddleOCR. The script degrades gracefully: any per-image OCR error is
# recorded and skipped, never fatal.
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def discover_keyframes(kf_root: Path, n: int, seed: int) -> list[Path]:
    """Reservoir-free uniform sample of N keyframe paths under kf_root."""
    all_imgs = [p for p in kf_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    rng = random.Random(seed)
    rng.shuffle(all_imgs)
    return sorted(all_imgs[:n])


def _build_easyocr(gpu: bool):
    import easyocr  # type: ignore[import-not-found]

    # Vietnamese + English; vi covers the diacritics, en catches latin scene text.
    return easyocr.Reader(["vi", "en"], gpu=gpu)


def _easyocr_text(reader, path: Path) -> tuple[str, int]:
    res = reader.readtext(str(path), detail=1, paragraph=False)
    texts = [t for (_box, t, _conf) in res if isinstance(t, str) and t.strip()]
    return " ".join(texts), len(texts)


def _build_paddle(gpu: bool):
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    return PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=gpu, show_log=False)


def _paddle_text(ocr, path: Path) -> tuple[str, int]:
    res = ocr.ocr(str(path), cls=True)
    lines = []
    for page in res or []:
        for entry in page or []:
            txt = entry[1][0] if entry and len(entry) > 1 else None
            if isinstance(txt, str) and txt.strip():
                lines.append(txt)
    return " ".join(lines), len(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample keyframes + OCR -> JSONL.")
    ap.add_argument("--kf-root", type=Path, default=Path("/tmp/aic2025/kf"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("/tmp/c1cal/ocr_sample.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=("easyocr", "paddle"), default="easyocr")
    ap.add_argument("--gpu", type=int, default=1, help="1=use GPU (default), 0=CPU")
    args = ap.parse_args()

    if not args.kf_root.is_dir():
        print(f"ERROR: no keyframe root {args.kf_root}", file=sys.stderr)
        return 2

    paths = discover_keyframes(args.kf_root, args.n, args.seed)
    print(f"[ocr] sampled {len(paths)} keyframes from {args.kf_root}", flush=True)
    if not paths:
        print("ERROR: no keyframes found", file=sys.stderr)
        return 3

    use_gpu = bool(args.gpu)
    if args.backend == "easyocr":
        try:
            reader = _build_easyocr(use_gpu)
        except Exception as exc:  # noqa: BLE001 - report + fall back to CPU
            print(f"[ocr] easyocr GPU init failed ({exc}); retrying CPU", flush=True)
            reader = _build_easyocr(False)
        run = lambda p: _easyocr_text(reader, p)  # noqa: E731
    else:
        ocr = _build_paddle(use_gpu)
        run = lambda p: _paddle_text(ocr, p)  # noqa: E731

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_text = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, p in enumerate(paths):
            try:
                text, n_boxes = run(p)
                ok = True
                err = None
            except Exception as exc:  # noqa: BLE001 - per-image errors are non-fatal
                text, n_boxes, ok, err = "", 0, False, str(exc)
            rec = {"path": str(p), "text": text, "n_boxes": n_boxes, "ok": ok}
            if err:
                rec["error"] = err
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += int(ok)
            n_text += int(bool(text.strip()))
            if (i + 1) % 25 == 0 or i == len(paths) - 1:
                el = time.time() - t0
                print(
                    f"[ocr] {i + 1}/{len(paths)} | ok={n_ok} with_text={n_text} "
                    f"| {el:.0f}s | {(i + 1) / el:.1f} img/s",
                    flush=True,
                )

    print(
        f"[ocr] DONE backend={args.backend} gpu={use_gpu} "
        f"ok={n_ok}/{len(paths)} with_text={n_text} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
