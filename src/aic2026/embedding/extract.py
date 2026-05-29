# Implements SPEC-0004 SS 3-4 (offline extraction).
"""Offline batch extraction: directory of images -> `.npy` + manifest.

Per ADR-0003 image-tower extraction runs **offline** on the GH200. The
output of this module is the input to SPEC-0006 (Milvus ingestion). The
file pair is:

- `<out>.npy`              float32 (n, dim)
- `<out>.manifest.jsonl`   one JSON object per line: {row, frame_id, path}

Row indices line up byte-for-byte with the matrix rows. `frame_id` is the
file stem; downstream specs join on `frame_id` to MockTask ground truth.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aic2026.embedding.base import Embedder

logger = logging.getLogger(__name__)

# SPEC-0004 SS 6 quality gate.
L2_NORM_TOLERANCE: float = 1e-3

# Discoverable image suffixes; sorted so traversal order is deterministic.
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class ExtractionResult:
    """Result handle returned by `extract_image_embeddings`."""

    n: int
    dim: int
    vectors_path: Path
    manifest_path: Path


def discover_images(input_dir: Path) -> list[Path]:
    """Sorted-by-path image files under `input_dir` (non-recursive shallow).

    Non-recursive is the slice-time default; deep walks are a follow-up
    when the real corpus directory shape is known.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input is not a directory: {input_dir}")
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _batched(items: list[Path], size: int) -> Iterable[list[Path]]:
    if size <= 0:
        raise ValueError(f"batch_size must be positive; got {size}")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def extract_image_embeddings(
    paths: list[Path],
    embedder: Embedder,
    *,
    out: Path,
    batch_size: int = 32,
) -> ExtractionResult:
    """Batch-encode `paths` through `embedder` and write `.npy` + manifest.

    `out` is a base path; the function writes `out.with_suffix('.npy')`
    and `out.parent / f"{out.stem}.manifest.jsonl"`.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vectors_path = out.with_suffix(".npy")
    manifest_path = out.parent / f"{out.stem}.manifest.jsonl"

    dim = embedder.dim
    matrix = np.zeros((len(paths), dim), dtype=np.float32)

    for batch_idx, batch in enumerate(_batched(paths, batch_size)):
        logger.info(
            "encoding batch %d (%d files) with %s",
            batch_idx,
            len(batch),
            embedder.model_id,
        )
        vecs = embedder.encode_image(batch)
        if vecs.shape != (len(batch), dim):
            raise ValueError(f"embedder returned shape {vecs.shape}; expected {(len(batch), dim)}")
        start = batch_idx * batch_size
        matrix[start : start + len(batch)] = vecs.astype(np.float32, copy=False)

    if matrix.size > 0:
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(np.abs(norms - 1.0) > L2_NORM_TOLERANCE):
            raise ValueError(
                "embedder returned rows that are not unit-norm "
                f"(max deviation {float(np.max(np.abs(norms - 1.0)))})"
            )

    np.save(vectors_path, matrix, allow_pickle=False)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row, p in enumerate(paths):
            fh.write(
                json.dumps(
                    {"row": row, "frame_id": p.stem, "path": str(p)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    return ExtractionResult(
        n=len(paths),
        dim=dim,
        vectors_path=vectors_path,
        manifest_path=manifest_path,
    )
