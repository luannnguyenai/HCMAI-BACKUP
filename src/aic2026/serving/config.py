# Implements SPEC-0026 SS 3 (ServingConfig).
"""Static configuration for the MVP serving API.

Frozen dataclass so the running service cannot mutate its own deployment
contract. Constructed once at boot (from CLI flags / env in `aic2026.cli.serve`)
and threaded into `create_app`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Default Milvus standalone endpoint on the shared MVP server (ADR-0014).
DEFAULT_MILVUS_URI: str = "http://127.0.0.1:19530"
DEFAULT_COLLECTION: str = "keyframes"
# The two online text-tower lanes (ADR-0003); qwen3vl is offline-only (ADR-0012).
DEFAULT_ONLINE_LANES: tuple[str, ...] = ("siglip2", "metaclip2")


@dataclass(frozen=True)
class ServingConfig:
    """Deployment contract for one serving process (SPEC-0026 SS 3)."""

    milvus_uri: str = DEFAULT_MILVUS_URI  # http://127.0.0.1:19530 (standalone, ADR-0014)
    collection: str = DEFAULT_COLLECTION
    online_lanes: tuple[str, ...] = DEFAULT_ONLINE_LANES
    thumb_root: Path = Path("/data/thumbs")  # hydrated from R2 (ADR-0015)
    full_root: Path = Path("/data/frames")
    github_repo: str | None = None  # "owner/repo" for issue capture; None -> local fallback
    issue_fallback_dir: Path = Path("/data/issues")  # local fallback sink (never lose a report)
    shared_secret: str | None = None  # shared-secret gate, required in prod (SS 9 Q1 RESOLVED)
    encode_device: str = "cpu"  # text-tower device; MVP runs CPU (SS 9 Q4 RESOLVED)
    encoder_dim_overrides: dict[str, int] = field(default_factory=dict)  # test/dev hook

    def __post_init__(self) -> None:
        # Normalise path-like fields so callers may pass strings.
        object.__setattr__(self, "thumb_root", Path(self.thumb_root))
        object.__setattr__(self, "full_root", Path(self.full_root))
        object.__setattr__(self, "issue_fallback_dir", Path(self.issue_fallback_dir))
