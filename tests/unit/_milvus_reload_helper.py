# Subprocess helper for SPEC-0006 AC9 (multi-vector cross-process reload).
"""Build / load a multi-vector Milvus Lite collection in a *fresh* process.

The milvus-lite 3.0 multi-vector reload regression (SPEC-0006 SS 12) only
surfaces when a collection persisted by one process is reopened *cold* by a
brand-new process (a fresh interpreter re-deserialising the on-disk segments) -
an in-process reopen keeps the correct segments in memory and hides the bug.
So the AC9 regression test drives this script as two sequential subprocesses:
`build` writes the .db, then `load` reopens it and searches each field.

Output protocol (stdout, one token per line):
  build: ``SKIP:<reason>`` or ``BUILD_OK``
  load:  ``SKIP:<reason>``; then ``LOAD_FAIL:<msg>`` or ``LOAD_OK`` followed by
         one ``<field>:<dim>:OK`` / ``<field>:<dim>:FAIL:<msg>`` line per field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Distinct dims so a cross-wired reload (every field served at the first
# field's dim) raises a dim mismatch instead of silently mis-ranking.
FIELDS: tuple[tuple[str, int], ...] = (("siglip2", 8), ("metaclip2", 4), ("qwen3vl", 6))
_N_FRAMES = 6
_COLLECTION = "keyframes"


def _store(uri: str):
    from aic2026.index.milvus_schema import DenseField
    from aic2026.index.milvus_store import MilvusKeyframeStore

    fields = tuple(DenseField(name, dim) for name, dim in FIELDS)
    return MilvusKeyframeStore(uri=uri, collection=_COLLECTION, fields=fields)


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    m = rng.standard_normal((n, d)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def _write_source(root: Path, name: str, dim: int, frame_ids: list[str], video: str):
    from aic2026.index.milvus_store import EncoderSource

    field_dir = root / name
    field_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash((name, video))) % (2**32))
    vectors = _unit(rng, len(frame_ids), dim)
    npy = field_dir / f"{video}.npy"
    np.save(npy, vectors)
    manifest = field_dir / f"{video}.manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row, fid in enumerate(frame_ids):
            fh.write(json.dumps({"row": row, "frame_id": fid, "path": f"{fid}.jpg"}) + "\n")
    return EncoderSource(vectors=npy, manifest=manifest)


def _init_errors() -> tuple[type[BaseException], ...]:
    from pymilvus.exceptions import MilvusException

    return (RuntimeError, OSError, MilvusException)


def build(uri: str, src_root: str) -> int:
    root = Path(src_root)
    frame_ids = [f"{i:04d}" for i in range(_N_FRAMES)]
    video = "L25_V011"
    try:
        store = _store(uri)
        store.ensure_collection()
        sources = {name: _write_source(root, name, dim, frame_ids, video) for name, dim in FIELDS}
        store.ingest(sources, video_id=video)
    except _init_errors() as exc:  # Lite cannot bind loopback (sandbox) -> skip.
        print(f"SKIP:{exc!r}")
        return 0
    print("BUILD_OK")
    return 0


def load(uri: str) -> int:
    import numpy as np

    try:
        store = _store(uri)
        # Query each field with a unit vector OF THAT FIELD'S DIM. On the buggy
        # engine the non-first fields reload at the first field's dim, so this
        # raises "loaded index dim .. != expected dim ..".
        rng = np.random.default_rng(123)
        first_field = FIELDS[0][0]
        # search() calls load_collection internally (the reload boundary).
        store.search(first_field, _unit(rng, 1, FIELDS[0][1])[0])
    except _init_errors() as exc:
        print(f"SKIP:{exc!r}")
        return 0
    except Exception as exc:  # report any load failure verbatim.
        print(f"LOAD_FAIL:{exc}")
        return 0
    print("LOAD_OK")
    for name, dim in FIELDS:
        try:
            hits = store.search(name, _unit(rng, 1, dim)[0], top_k=2)
            ok = len(hits) == 1 and len(hits[0]) >= 1
            print(f"{name}:{dim}:{'OK' if ok else 'FAIL:empty'}")
        except Exception as exc:
            print(f"{name}:{dim}:FAIL:{exc}")
    return 0


def main() -> int:
    mode = sys.argv[1]
    if mode == "build":
        return build(sys.argv[2], sys.argv[3])
    if mode == "load":
        return load(sys.argv[2])
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
