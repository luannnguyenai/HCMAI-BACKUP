# GT-free eval for the SPEC-0006 Milvus keyframe store (latency + recall + qualitative).
#
# Not committed application code; an on-box harness. Encodes ~N KIS query texts
# with each floor encoder, runs per-field top-k ANN against the live Milvus
# collection, and reports:
#   - per-lane query latency p50/p95 (H200 proxy, not the finals 5070 number)
#   - recall@k of the HNSW path vs an exact numpy IP baseline (sweep over ef)
#   - top-5 frame_ids per query (qualitative; json + html)
#
# Vectors are the SPEC-0004 producer outputs (L2-normalised), so IP == cosine.
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from aic2026.index.milvus_schema import FLOOR_FIELDS
from aic2026.index.milvus_store import MilvusKeyframeStore


def load_queries(query_dir: Path, n: int) -> list[str]:
    files = sorted(query_dir.rglob("*kis*.txt")) or sorted(query_dir.rglob("*.txt"))
    out: list[str] = []
    for p in files:
        txt = " ".join(p.read_text(encoding="utf-8", errors="replace").split())
        if txt:
            out.append(txt)
        if len(out) >= n:
            break
    return out


def build_encoder(name: str, *, device: str, dtype: str, qwen_impl_src: str | None):
    name = name.lower()
    if name == "siglip2":
        from aic2026.embedding.siglip2 import SigLip2Embedder

        return SigLip2Embedder(device=device, dtype=dtype)
    if name == "metaclip2":
        from aic2026.embedding.metaclip2 import MetaClip2Embedder

        return MetaClip2Embedder(device=device, dtype=dtype)
    if name == "qwen3vl":
        from aic2026.embedding.qwen3vl_embed import Qwen3VLEmbedder

        return Qwen3VLEmbedder(device=device, dtype=dtype, impl_src=qwen_impl_src)
    raise ValueError(f"unknown encoder {name!r}")


def l2(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (mat / n).astype(np.float32, copy=False)


def load_lane_matrix(index_root: Path, lane: str) -> tuple[np.ndarray, list[str]]:
    """Stack one lane's per-video npy in sorted-video manifest order.

    Builds the GLOBAL primary key `<video>_<frame_id>` (matching the SPEC-0006
    store's composed pk), since the SPEC-0004 manifest `frame_id` is per-video
    and repeats across videos. No manifest rewrite (validates the Part 1 fix).
    """
    lane_dir = index_root / lane
    videos = sorted(p.stem for p in lane_dir.glob("*.npy"))
    mats: list[np.ndarray] = []
    pks: list[str] = []
    for v in videos:
        mats.append(np.load(lane_dir / f"{v}.npy"))
        with (lane_dir / f"{v}.manifest.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    pks.append(f"{v}_{json.loads(line)['frame_id']}")
    matrix = np.concatenate(mats, axis=0).astype(np.float32, copy=False)
    if matrix.shape[0] != len(pks):
        raise SystemExit(f"{lane}: matrix rows {matrix.shape[0]} != manifest ids {len(pks)}")
    return matrix, pks


def exact_topk(query: np.ndarray, matrix: np.ndarray, fids: list[str], k: int) -> list[str]:
    scores = matrix @ query  # (N,)
    idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [fids[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--index-root", required=True)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--collection", default="keyframes")
    ap.add_argument("--fields", default="siglip2,metaclip2,qwen3vl")
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--ef-latency", type=int, default=256)
    ap.add_argument("--ef-sweep", default="128,256,512")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--qwen-impl-src", default=None)
    args = ap.parse_args()

    index_root = Path(args.index_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lanes = [s.strip() for s in args.fields.split(",") if s.strip()]
    ef_sweep = [int(x) for x in args.ef_sweep.split(",") if x.strip()]
    queries = load_queries(Path(args.query_dir), args.n_queries)
    print(f"loaded {len(queries)} queries; lanes={lanes}", flush=True)

    by_name = {f.name: f for f in FLOOR_FIELDS}
    store = MilvusKeyframeStore(uri=args.uri, collection=args.collection)

    report: dict[str, object] = {
        "uri": args.uri,
        "collection": args.collection,
        "n_queries": len(queries),
        "top_k": args.top_k,
        "ef_latency": args.ef_latency,
        "lanes": {},
    }
    qualitative: dict[str, object] = {"queries": queries, "lanes": {}}

    for lane in lanes:
        is_lite = store.is_lite
        print(f"\n=== lane {lane} (dim {by_name[lane].dim}) ===", flush=True)
        enc = build_encoder(
            lane, device=args.device, dtype=args.dtype, qwen_impl_src=args.qwen_impl_src
        )
        q = np.asarray(enc.encode_text(queries), dtype=np.float32)
        q = l2(q)
        del enc
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        # warm-up (triggers collection load; excluded from timing)
        store.search(lane, q[0:1], top_k=args.top_k, ef_search=args.ef_latency)

        # latency at ef_latency
        lat_ms: list[float] = []
        hnsw_hits: list[list] = []
        for i in range(len(queries)):
            t0 = time.perf_counter()
            res = store.search(lane, q[i : i + 1], top_k=args.top_k, ef_search=args.ef_latency)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            hnsw_hits.append(res[0])
        p50 = statistics.median(lat_ms)
        p95 = sorted(lat_ms)[max(0, round(0.95 * len(lat_ms)) - 1)]
        print(f"  latency ef={args.ef_latency}: p50={p50:.2f}ms p95={p95:.2f}ms", flush=True)

        # exact baseline + recall sweep (skip recall for Lite/FLAT: exact already).
        # Two recall flavours, because consecutive keyframes are near-duplicates
        # so the score at rank k is often tied across several frames:
        #   recall_set   = |HNSW ids INTERSECT exact ids| / k  (tie-sensitive)
        #   recall_score = |HNSW hits with score >= exact k-th score - eps| / k
        #                  (tie-robust: "did HNSW find frames as good as the true cutoff")
        recall_set_by_ef: dict[str, float] = {}
        recall_score_by_ef: dict[str, float] = {}
        tie_mult: list[int] = []
        if not is_lite:
            eps = 1e-5
            matrix, pks = load_lane_matrix(index_root, lane)
            exact_ids: list[set[str]] = []
            tau: list[float] = []  # exact k-th score per query
            for i in range(len(queries)):
                scores = matrix @ q[i]
                idx = np.argpartition(-scores, min(args.top_k, len(scores) - 1))[: args.top_k]
                order = idx[np.argsort(-scores[idx])]
                exact_ids.append({pks[j] for j in order})
                kth = float(scores[order[-1]])
                tau.append(kth)
                tie_mult.append(int(np.sum(scores >= kth - eps)))
            del matrix
            for ef in ef_sweep:
                set_recs: list[float] = []
                score_recs: list[float] = []
                for i in range(len(queries)):
                    res = store.search(lane, q[i : i + 1], top_k=args.top_k, ef_search=ef)
                    got_ids = {h.pk for h in res[0]}
                    set_recs.append(len(got_ids & exact_ids[i]) / float(args.top_k))
                    n_good = sum(1 for h in res[0] if h.score >= tau[i] - eps)
                    score_recs.append(n_good / float(args.top_k))
                recall_set_by_ef[str(ef)] = round(float(np.mean(set_recs)), 4)
                recall_score_by_ef[str(ef)] = round(float(np.mean(score_recs)), 4)
                print(
                    f"  ef={ef}: recall_set@{args.top_k}={recall_set_by_ef[str(ef)]} "
                    f"recall_score@{args.top_k}={recall_score_by_ef[str(ef)]}",
                    flush=True,
                )
            print(f"  boundary tie multiplicity @k (mean): {np.mean(tie_mult):.1f}", flush=True)
        else:
            recall_set_by_ef = {"flat": 1.0}
            recall_score_by_ef = {"flat": 1.0}
            print("  Lite/FLAT: recall exact by construction", flush=True)

        report["lanes"][lane] = {
            "dim": by_name[lane].dim,
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_all_ms": [round(x, 2) for x in lat_ms],
            "recall_set_by_ef": recall_set_by_ef,
            "recall_score_by_ef": recall_score_by_ef,
            "boundary_tie_mult_mean": round(float(np.mean(tie_mult)), 1) if tie_mult else None,
        }
        qualitative["lanes"][lane] = [
            [
                {
                    "rank": h.rank,
                    "pk": h.pk,
                    "frame_id": h.frame_id,
                    "video_id": h.video_id,
                    "score": round(h.score, 4),
                }
                for h in hits[:5]
            ]
            for hits in hnsw_hits
        ]

    (out_dir / "latency_recall.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "qualitative.json").write_text(json.dumps(qualitative, indent=2), encoding="utf-8")
    _write_html(out_dir / "qualitative.html", queries, qualitative["lanes"], lanes)
    print(f"\nwrote {out_dir}/latency_recall.json, qualitative.json, qualitative.html", flush=True)


def _write_html(path: Path, queries: list[str], lanes_q: dict, lanes: list[str]) -> None:
    rows = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:sans-serif;font-size:13px}",
        "table{border-collapse:collapse;margin:8px 0}",
        "td,th{border:1px solid #ccc;padding:3px 6px;vertical-align:top}",
        "th{background:#eee}.q{font-weight:bold;background:#f6f6f6}</style></head><body>",
        "<h2>SPEC-0006 Milvus proxy: qualitative top-5</h2>",
    ]
    for qi, qtext in enumerate(queries):
        rows.append(f"<div class='q'>Q{qi}: {qtext[:300]}</div>")
        rows.append("<table><tr>" + "".join(f"<th>{ln}</th>" for ln in lanes) + "</tr><tr>")
        for ln in lanes:
            cell = "<br>".join(
                f"{h['rank']}. {h['pk']} ({h['score']:+.3f})" for h in lanes_q[ln][qi]
            )
            rows.append(f"<td>{cell}</td>")
        rows.append("</tr></table>")
    rows.append("</body></html>")
    path.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
