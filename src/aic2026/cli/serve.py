# Implements SPEC-0026 SS 3 (bin/serve CLI surface).
"""`bin/serve` - run the MVP serving API (SPEC-0026).

Builds a `ServingConfig` from flags + env, constructs the FastAPI app (real
Milvus standalone store + real SPEC-0004 text towers), and runs it under
uvicorn. The shared secret is read from the environment (`AIC2026_SHARED_SECRET`)
rather than a flag so it does not appear in the process list.

Requires the `serving` + `index` extras and, for the real text towers, the
`embedding` extra:
    uv sync --extra serving --extra index --extra embedding
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from aic2026.serving.config import (
    DEFAULT_COLLECTION,
    DEFAULT_MILVUS_URI,
    DEFAULT_ONLINE_LANES,
    ServingConfig,
)

app = typer.Typer(
    add_completion=False,
    help=(
        "AIC2026 MVP serving API (SPEC-0026). FastAPI + WebSocket over the "
        "merged SPEC-0006 MilvusBackend: Vietnamese KIS query, frame detail, "
        "static keyframe images, issue capture, health/readiness. Requires the "
        "`serving` + `index` (+ `embedding` for real towers) extras."
    ),
)

SHARED_SECRET_ENV: str = "AIC2026_SHARED_SECRET"
EXIT_OK = 0


def _logger() -> logging.Logger:
    return logging.getLogger("aic2026.cli.serve")


@app.command()
def run(
    host: Annotated[str, typer.Option(help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    milvus_uri: Annotated[
        str, typer.Option(help="Milvus standalone endpoint (ADR-0014).")
    ] = DEFAULT_MILVUS_URI,
    collection: Annotated[str, typer.Option(help="Milvus collection name.")] = DEFAULT_COLLECTION,
    lanes: Annotated[str, typer.Option(help="CSV of online lanes.")] = ",".join(
        DEFAULT_ONLINE_LANES
    ),
    thumb_root: Annotated[Path, typer.Option(help="Local thumbnail tier root (ADR-0015).")] = Path(
        "/data/thumbs"
    ),
    full_root: Annotated[Path, typer.Option(help="Local full-image root.")] = Path("/data/frames"),
    github_repo: Annotated[
        str | None, typer.Option(help="'owner/repo' for issue capture; omit for local fallback.")
    ] = None,
    issue_fallback_dir: Annotated[Path, typer.Option(help="Local issue fallback sink.")] = Path(
        "/data/issues"
    ),
    encode_device: Annotated[
        str, typer.Option(help="Text-tower device: cpu (MVP) or cuda.")
    ] = "cpu",
) -> None:
    """Serve the MVP API (blocking). Ctrl-C to stop."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised manually
        typer.secho(
            f"ERROR: {exc}. Install the serving extra: `uv sync --extra serving`.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(2) from None

    from aic2026.serving.app import create_app

    online_lanes = tuple(n.strip() for n in lanes.split(",") if n.strip())
    config = ServingConfig(
        milvus_uri=milvus_uri,
        collection=collection,
        online_lanes=online_lanes,
        thumb_root=thumb_root,
        full_root=full_root,
        github_repo=github_repo,
        issue_fallback_dir=issue_fallback_dir,
        shared_secret=os.environ.get(SHARED_SECRET_ENV),
        encode_device=encode_device,
    )
    if config.shared_secret is None:
        _logger().warning(
            "%s is not set: the API is running WITHOUT a shared-secret gate. "
            "Set it in production (SPEC-0026 SS 9 Q1).",
            SHARED_SECRET_ENV,
        )
    application = create_app(config)
    uvicorn.run(application, host=host, port=port)


def main() -> None:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
