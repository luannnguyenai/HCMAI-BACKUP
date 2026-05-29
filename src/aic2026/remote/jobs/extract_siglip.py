# Implements SPEC-0022 SS 3 (jobs.extract-siglip) and supports AC6.
"""SigLIP-2 image-embedding extraction job.

Wraps SPEC-0004's `SigLip2Embedder` + `extract_image_embeddings`. The job
walks `--config input_dir=...`, encodes each image, writes `v.npy` and
`v.manifest.jsonl` under `RunContext.local_run_dir/`. The runner then
uploads that directory to R2 wholesale.

This module imports `aic2026.embedding.siglip2` *lazily* inside the job
function so the registry import path stays free of `torch`. SPEC-0004 already
guarantees `SigLip2Embedder` lazy-imports torch; this layer just reinforces
the discipline by not pulling that module at decoration time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aic2026.remote.context import RunContext
from aic2026.remote.registry import register

# SPEC-0004 SS 3 NFR: SigLIP-2 So400m/16@384 is fixed at 1024-d.
DEFAULT_BATCH_SIZE: int = 32


@register("extract-siglip")
def extract_siglip(ctx: RunContext, config: dict[str, Any]) -> None:
    """Run SigLIP-2 image-tower extraction over `input_dir`.

    Required config:
        input_dir (str | Path): directory of `*.jpg/jpeg/png/webp` keyframes.

    Optional config:
        batch_size (int): default 32.
        device (str): "cpu" | "cuda" (default cuda when available).
        dtype (str): "float16" (default) | "float32".

    Outputs (written under `ctx.local_run_dir/`):
        v.npy
        v.manifest.jsonl
    """
    # Lazy imports: torch / open_clip / pillow only load when this job runs.
    from aic2026.embedding.extract import discover_images, extract_image_embeddings
    from aic2026.embedding.siglip2 import SigLip2Embedder

    input_dir_raw = config.get("input_dir")
    if not input_dir_raw:
        raise ValueError(
            "extract-siglip requires `--config input_dir=<path>` "
            "pointing at a directory of *.jpg/jpeg/png/webp files."
        )
    input_dir = Path(input_dir_raw).expanduser()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    device = str(config.get("device", "cuda"))
    dtype = str(config.get("dtype", "float16"))

    paths = discover_images(input_dir)
    if not paths:
        # An empty run is still a valid run - the spec promises a zero-row
        # .npy + empty manifest; we let `extract_image_embeddings` handle it.
        pass

    ctx.local_run_dir.mkdir(parents=True, exist_ok=True)
    embedder = SigLip2Embedder(device=device, dtype=dtype)
    extract_image_embeddings(
        paths,
        embedder,
        out=ctx.local_run_dir / "v",
        batch_size=batch_size,
    )
