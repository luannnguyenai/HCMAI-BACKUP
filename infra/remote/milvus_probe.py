# Read-only diagnostic: is the keyframes HNSW index built, and does Milvus
# search match a numpy exact-IP baseline on the SAME stored vector? Resolves
# whether recall<1 is approximation (HNSW) or a vector/order mismatch.
from __future__ import annotations

import sys

import numpy as np
from pymilvus import MilvusClient

sys.path.insert(0, "/tmp")
from milvus_eval import load_lane_matrix  # noqa: E402
from pathlib import Path  # noqa: E402

URI = "http://127.0.0.1:19530"
LANE = "siglip2"
ROOT = Path("/tmp/aic2025/index_milvus")

c = MilvusClient(uri=URI)
c.load_collection("keyframes")

print("indexes:", c.list_indexes("keyframes"))
try:
    print("describe siglip2 index:", c.describe_index("keyframes", LANE))
except Exception as e:
    print("describe_index err:", e)

# one stored vector as the query (exact self-known)
row = c.query("keyframes", filter="video_id == 'L25_V001'",
              output_fields=["frame_id", LANE], limit=1)[0]
qv = np.asarray(row[LANE], dtype=np.float32)
qn = qv / (np.linalg.norm(qv) or 1.0)


def milvus_topk(ef: int) -> list[str]:
    r = c.search("keyframes", data=[qv.tolist()], anns_field=LANE, limit=200,
                 output_fields=["frame_id"],
                 search_params={"metric_type": "IP", "params": {"ef": ef}})
    return [h["entity"]["frame_id"] for h in r[0]]


matrix, fids = load_lane_matrix(ROOT, LANE)
print("matrix", matrix.shape, "ids", len(fids))
# numpy exact over RAW stored vectors (as ingested) and over L2-normed
sc_raw = matrix @ qv
ex_raw = {fids[i] for i in np.argsort(-sc_raw)[:200]}
mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
sc_norm = mn @ qn
ex_norm = {fids[i] for i in np.argsort(-sc_norm)[:200]}

for ef in (8, 128, 2048):
    got = set(milvus_topk(ef))
    print(f"ef={ef:>4} | overlap vs exact_raw={len(got & ex_raw):3d}/200 "
          f"vs exact_norm={len(got & ex_norm):3d}/200")

# are stored vectors actually unit norm?
norms = np.linalg.norm(matrix[:5], axis=1)
print("sample stored norms:", [round(float(x), 4) for x in norms])
