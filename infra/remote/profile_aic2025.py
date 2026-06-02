#!/usr/bin/env python3
"""Profile the AIC2025 proxy corpus (research-note 07).

Walks a downloaded copy of the AIC2025 Drive folder and reports ground-truth
structure we can only *estimate* from the Drive listing: per-collection keyframe
counts, the frame-naming scheme, a resolution/file-size histogram, the query-set
format + language, and a video inventory. Writes a JSON report and prints a
human summary.

This is deliberately dependency-light so it runs on a fresh GPU box:
  * Pillow (`PIL`) is used for image resolution **only if importable**; absent
    -> resolution stats are skipped (counts/sizes still reported).
  * `ffprobe` is shelled out for video duration/codec **only if on PATH**.
Everything else is stdlib.

Usage:
    python infra/remote/profile_aic2025.py --root /tmp/aic2025 \
        [--out /tmp/aic2025_profile.json] [--sample-images 500] [--sample-videos 20]

The profiler operates on whatever is already on disk under --root (decoupled
from how it was downloaded). To fetch the Drive folder first:
    uv run pip install gdown
    gdown --folder "<drive-folder-url>" -O /tmp/aic2025
    # then unzip the keyframe archives you want to profile, e.g.:
    #   for z in /tmp/aic2025/Keyframes_*.zip; do unzip -q "$z" -d /tmp/aic2025/kf; done
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
TEXT_EXTS = {".txt", ".csv", ".json", ".jsonl", ".tsv"}

# A keyframe path usually looks like ".../L25/V001/123.jpg" or ".../L25_V001_123.jpg".
# We infer the collection token (L\d+) and a per-frame index to sanity-check density.
_COLLECTION_RE = re.compile(r"(L\d{1,3})", re.IGNORECASE)


def _vietnamese_diacritic_ratio(text: str) -> float:
    """Fraction of alpha chars carrying a Vietnamese diacritic (NFD combining
    mark or the đ/Đ stroke). A cheap "is this Vietnamese" / tone-density signal."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    marked = 0
    for ch in alpha:
        if ch in ("đ", "Đ"):
            marked += 1
            continue
        if any(unicodedata.combining(d) for d in unicodedata.normalize("NFD", ch)):
            marked += 1
    return marked / len(alpha)


def _fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def _iter_files(root: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _collection_of(path: Path, root: Path) -> str:
    """Best-effort collection token (e.g. 'L25') from a keyframe path; '?' if none."""
    rel = path.relative_to(root).as_posix()
    m = _COLLECTION_RE.search(rel)
    return m.group(1).upper() if m else "?"


def profile_keyframes(root: Path, sample_images: int) -> dict[str, Any]:
    imgs = _iter_files(root, IMAGE_EXTS)
    if not imgs:
        return {"present": False, "note": "no image files found under --root"}

    per_collection: Counter[str] = Counter()
    sizes: list[int] = []
    for p in imgs:
        per_collection[_collection_of(p, root)] += 1
        sizes.append(p.stat().st_size)

    # Resolution histogram on a stride-sampled subset (Pillow optional).
    res_hist: Counter[str] = Counter()
    res_note = ""
    try:
        from PIL import Image

        stride = max(1, len(imgs) // max(1, sample_images))
        sampled = imgs[::stride][:sample_images]
        for p in sampled:
            try:
                with Image.open(p) as im:
                    res_hist[f"{im.width}x{im.height}"] += 1
            except Exception as exc:
                res_hist[f"<unreadable:{type(exc).__name__}>"] += 1
        res_note = f"sampled {len(sampled)} of {len(imgs)} frames (stride={stride})"
    except ImportError:
        res_note = "Pillow not installed; resolution skipped (pip install pillow)"

    sample_paths = [p.relative_to(root).as_posix() for p in imgs[:5]]
    return {
        "present": True,
        "total_frames": len(imgs),
        "per_collection": dict(sorted(per_collection.items())),
        "filesize_bytes": {
            "total": sum(sizes),
            "total_human": _fmt_bytes(sum(sizes)),
            "mean": int(statistics.mean(sizes)),
            "median": int(statistics.median(sizes)),
            "min": min(sizes),
            "max": max(sizes),
        },
        "resolution_hist": dict(res_hist.most_common(20)),
        "resolution_note": res_note,
        "sample_paths": sample_paths,
    }


def _load_text_lines(p: Path) -> list[str]:
    """Best-effort load of query-like text. Handles .txt/.tsv/.csv (per-line),
    and .json/.jsonl (extract string leaves heuristically)."""
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    suffix = p.suffix.lower()
    if suffix in (".json", ".jsonl"):
        out: list[str] = []
        for chunk in raw.splitlines() if suffix == ".jsonl" else [raw]:
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            out.extend(_json_strings(obj))
        return out
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _json_strings(obj: Any) -> list[str]:
    """Collect non-trivial string leaves from a parsed JSON object."""
    found: list[str] = []
    if isinstance(obj, str):
        if len(obj) >= 4:
            found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_json_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_json_strings(v))
    return found


def profile_queries(query_dir: Path) -> dict[str, Any]:
    if not query_dir.is_dir():
        return {"present": False, "note": f"no query dir at {query_dir}"}
    files = _iter_files(query_dir, TEXT_EXTS)
    ext_hist = Counter(p.suffix.lower() for p in files)
    strings: list[str] = []
    for p in files:
        strings.extend(_load_text_lines(p))

    if not strings:
        return {
            "present": True,
            "files": len(files),
            "ext_hist": dict(ext_hist),
            "note": "no text strings parsed (format may be non-text or nested)",
        }

    char_lens = [len(s) for s in strings]
    word_lens = [len(s.split()) for s in strings]
    dia = [_vietnamese_diacritic_ratio(s) for s in strings]
    return {
        "present": True,
        "files": len(files),
        "ext_hist": dict(ext_hist),
        "n_strings": len(strings),
        "char_len": {
            "mean": round(statistics.mean(char_lens), 1),
            "median": statistics.median(char_lens),
            "min": min(char_lens),
            "max": max(char_lens),
        },
        "word_len": {
            "mean": round(statistics.mean(word_lens), 1),
            "median": statistics.median(word_lens),
            "max": max(word_lens),
        },
        "vietnamese_diacritic_ratio_mean": round(statistics.mean(dia), 3),
        "fraction_with_any_diacritic": round(sum(1 for d in dia if d > 0) / len(dia), 3),
        "samples": strings[:8],
    }


def _ffprobe_duration(path: Path) -> dict[str, Any] | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_name,width,height",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        data = json.loads(out.stdout or "{}")
        dur = float(data.get("format", {}).get("duration", 0.0))
        streams = data.get("streams", [{}])
        return {"duration_s": round(dur, 1), "codec": streams[0].get("codec_name")}
    except (subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
        return None


def profile_videos(root: Path, sample_videos: int) -> dict[str, Any]:
    vids = _iter_files(root, VIDEO_EXTS)
    if not vids:
        return {"present": False, "note": "no video files found under --root"}
    ext_hist = Counter(p.suffix.lower() for p in vids)
    sizes = [p.stat().st_size for p in vids]
    probed: list[dict[str, Any]] = []
    stride = max(1, len(vids) // max(1, sample_videos))
    for p in vids[::stride][:sample_videos]:
        info = _ffprobe_duration(p)
        if info is not None:
            probed.append({"path": p.name, **info})
    return {
        "present": True,
        "total_videos": len(vids),
        "ext_hist": dict(ext_hist),
        "filesize_total_human": _fmt_bytes(sum(sizes)),
        "ffprobe_note": "ffprobe not on PATH; durations skipped" if not probed else "",
        "probed_sample": probed,
    }


def profile_archives(root: Path) -> dict[str, Any]:
    zips = _iter_files(root, {".zip"})
    return {
        "count": len(zips),
        "archives": [{"name": p.name, "size_human": _fmt_bytes(p.stat().st_size)} for p in zips],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile the AIC2025 proxy corpus.")
    ap.add_argument("--root", type=Path, required=True, help="Downloaded corpus root.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    ap.add_argument(
        "--query-dir", type=Path, default=None, help="Override query dir (default <root>/query)."
    )
    ap.add_argument(
        "--sample-images", type=int, default=500, help="Frames to sample for resolution."
    )
    ap.add_argument("--sample-videos", type=int, default=20, help="Videos to ffprobe.")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"--root {root} is not a directory")
    query_dir = args.query_dir or (root / "query")

    report: dict[str, Any] = {
        "root": str(root),
        "archives": profile_archives(root),
        "keyframes": profile_keyframes(root, args.sample_images),
        "queries": profile_queries(query_dir),
        "videos": profile_videos(root, args.sample_videos),
    }

    print("=" * 72)
    print(f"AIC2025 proxy-corpus profile  root={root}")
    print("=" * 72)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out is not None:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
