# On-box adapter: make SPEC-0004 proxy manifests ingestible by the SPEC-0006 store.
#
# Finding (record in SPEC-0006 SS 10): the proxy SPEC-0004 manifests store
# `frame_id` as the per-video frame number ("001"), with the video identity
# living in the npy/manifest *filename* (L25_V001) and the `path`. But the
# SPEC-0006 store uses `frame_id` as the global Milvus primary key and parses
# `video_id` from it via the `L<NN>_V<NNN>` regex. As-is that (a) crashes in
# parse_video_id and (b) would collide PKs across videos ("001" repeats).
#
# This builds a sibling tree `index_milvus/<lane>/` that SYMLINKS the unchanged
# .npy vectors and rewrites each manifest so frame_id = "<video>_<frame>"
# (e.g. L25_V001_001): globally unique and parseable. Vectors are untouched;
# only the manifest id strings change. Idempotent.
from __future__ import annotations

import json
import os
from pathlib import Path

SRC = Path("/tmp/aic2025/index")
DST = Path("/tmp/aic2025/index_milvus")
LANES = ("siglip2", "metaclip2", "qwen3vl")


def main() -> None:
    total_frames = 0
    nvid = 0
    for lane in LANES:
        (DST / lane).mkdir(parents=True, exist_ok=True)
        for npy in sorted((SRC / lane).glob("*.npy")):
            vid = npy.stem
            link = DST / lane / f"{vid}.npy"
            if not link.exists():
                os.symlink(npy.resolve(), link)
            rows = []
            with (SRC / lane / f"{vid}.manifest.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    r["frame_id"] = f"{vid}_{r['frame_id']}"
                    rows.append(r)
            with (DST / lane / f"{vid}.manifest.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            if lane == "siglip2":
                total_frames += len(rows)
                nvid += 1
    print(f"videos={nvid} siglip2_frames={total_frames}")
    sample = (DST / "siglip2" / "L25_V001.manifest.jsonl").read_text(encoding="utf-8")
    print("sample:", sample.splitlines()[0])


if __name__ == "__main__":
    main()
