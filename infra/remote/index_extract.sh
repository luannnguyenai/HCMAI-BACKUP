#!/usr/bin/env bash
# Implements SPEC-0004 SS 3-4 (offline extraction) + ADR-0012 (offline
# visual-document lane). Offline INDEX extraction over the AIC2025 proxy
# keyframes for THREE encoders - siglip2 + metaclip2 + qwen3vl - writing a
# per-video .npy + .manifest.jsonl pair, then banking the whole tree to
# Cloudflare R2 (ADR-0011) so it survives lease rollover.
#
#   scp src/aic2026/embedding/metaclip2.py src/aic2026/embedding/qwen3vl_embed.py \
#       aic2026-gpu:aic2026/HEAD/src/aic2026/embedding/
#   scp infra/remote/index_extract.sh aic2026-gpu:.
#   ssh aic2026-gpu 'mkdir -p /tmp/aic2025 && CUDA_VISIBLE_DEVICES=1 \
#       setsid nohup bash index_extract.sh > /tmp/aic2025/index_extract.log 2>&1 &'
#
# Why a single Python process per encoder (not `./bin/embed images` per dir):
# `discover_images` is shallow/non-recursive (SPEC-0004 SS 2.2) and keyframes
# are nested per-video (<kf_root>/Keyframes_L25/keyframes/L25_V011/001.jpg), so
# a single `--input <kf_root>` finds nothing. We loop over each per-video subdir
# - but load each model ONCE and reuse it across all 546 videos, instead of
# paying the model-load cost 546x per encoder. This reuses the committed library
# (discover_images + extract_image_embeddings + the three Embedder classes)
# verbatim; no committed Python is changed.
#
# Args / env (all overridable):
#   DIR         project dir w/ the `embedding` extra synced (default ~/aic2026/HEAD)
#   KF_ROOT     keyframe root; per-video subdirs nested below (default /tmp/aic2025/kf)
#   OUT_ROOT    output base; writes <OUT_ROOT>/<enc>/<video>.npy + .manifest.jsonl
#               (default /tmp/aic2025/index)
#   ENCODERS    csv of encoders (default siglip2,metaclip2,qwen3vl)
#   QWEN_SRC    cloned QwenLM/Qwen3-VL-Embedding repo for qwen3vl --impl-src
#               (default ~/Qwen3-VL-Embedding)
#   QWEN_MODEL  HF repo id for qwen3vl backbone (default: the Qwen3VLEmbedder
#               built-in 2B repo). Set to Qwen/Qwen3-VL-Embedding-8B for the 8B lane.
#   BATCH_QWEN  override the qwen3vl per-call batch size (default 32; 8B wants ~8)
#   BANK        "1" to auto-upload OUT_ROOT to R2 at the end. Defaults to 1 for a
#               monolithic run (SHARD_COUNT=1) and 0 for shards (the bank watcher
#               banks once after all shards finish).
#   R2_PREFIX   R2 destination prefix (default index/<UTC-timestamp>)
#
# Parallel sharding (avoids GPU collisions; finishes 6x faster on 6 GPUs):
#   SHARD_COUNT  number of disjoint shards (default 1 = legacy monolithic run)
#   SHARD_INDEX  this shard's index in [0, SHARD_COUNT) (default 0); a shard owns
#                the videos where (sorted_video_index % SHARD_COUNT == SHARD_INDEX)
#   GPU          if set, pins this process to CUDA_VISIBLE_DEVICES=$GPU
#   SENTINEL_DIR where each shard drops index_shard_<k>.done on success / .fail on
#                error so the bank watcher knows when all shards are finished
#                (default /tmp/aic2025)
#
# Native dims: siglip2 1152, metaclip2 1024, qwen3vl 2048 (no MRL truncation).

set -u
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

DIR="${DIR:-$HOME/aic2026/HEAD}"
KF_ROOT="${KF_ROOT:-/tmp/aic2025/kf}"
OUT_ROOT="${OUT_ROOT:-/tmp/aic2025/index}"
ENCODERS="${ENCODERS:-siglip2,metaclip2,qwen3vl}"
QWEN_SRC="${QWEN_SRC:-$HOME/Qwen3-VL-Embedding}"
QWEN_MODEL="${QWEN_MODEL:-}"
BATCH_QWEN="${BATCH_QWEN:-}"
R2_PREFIX="${R2_PREFIX:-index/$(date -u +%Y%m%dT%H%M%SZ)}"

# --- parallel sharding (strict GPU partition; default = legacy monolithic run) ---
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
GPU="${GPU:-}"
SENTINEL_DIR="${SENTINEL_DIR:-/tmp/aic2025}"
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"
# Banking default: monolithic run banks itself; shards defer to the bank watcher.
if [ "$SHARD_COUNT" -gt 1 ]; then
  BANK="${BANK:-0}"
else
  BANK="${BANK:-1}"
fi

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH"; exit 1; }
[ -d "$DIR" ] || { echo "ERROR: no project dir $DIR"; exit 1; }
[ -d "$KF_ROOT" ] || { echo "ERROR: no keyframe root $KF_ROOT"; exit 1; }

# --- Qwen3-VL-Embedding official repo (its .process() API, not AutoModel) ---
if printf '%s' "$ENCODERS" | grep -q qwen3vl; then
  if [ ! -d "$QWEN_SRC" ]; then
    echo "== cloning QwenLM/Qwen3-VL-Embedding -> $QWEN_SRC =="
    git clone --depth 1 https://github.com/QwenLM/Qwen3-VL-Embedding "$QWEN_SRC" || true
  fi
  cd "$DIR" && uv pip install -q qwen-vl-utils 2>/dev/null || true
  echo "== qwen impl src: $QWEN_SRC =="
fi

cd "$DIR" || exit 1
mkdir -p "$OUT_ROOT"
echo "== index_extract start $(date -u +%FT%TZ) =="
echo "   DIR=$DIR"
echo "   KF_ROOT=$KF_ROOT  OUT_ROOT=$OUT_ROOT"
echo "   ENCODERS=$ENCODERS  GPU(CUDA_VISIBLE_DEVICES)=${CUDA_VISIBLE_DEVICES:-all}"
echo "   SHARD_INDEX=$SHARD_INDEX / SHARD_COUNT=$SHARD_COUNT  QWEN_MODEL=${QWEN_MODEL:-<default-2B>}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader 2>/dev/null || true

KF_ROOT="$KF_ROOT" OUT_ROOT="$OUT_ROOT" ENCODERS="$ENCODERS" QWEN_SRC="$QWEN_SRC" \
QWEN_MODEL="$QWEN_MODEL" BATCH_QWEN="$BATCH_QWEN" \
SHARD_COUNT="$SHARD_COUNT" SHARD_INDEX="$SHARD_INDEX" \
uv run python - <<'PY'
import os
import time
from pathlib import Path

from aic2026.embedding.extract import discover_images, extract_image_embeddings

kf_root = Path(os.environ["KF_ROOT"])
out_root = Path(os.environ["OUT_ROOT"])
encoders = [e.strip() for e in os.environ["ENCODERS"].split(",") if e.strip()]
qwen_src = os.environ["QWEN_SRC"]
qwen_model = os.environ.get("QWEN_MODEL") or None
batch_qwen = os.environ.get("BATCH_QWEN") or None
shard_count = int(os.environ.get("SHARD_COUNT", "1"))
shard_index = int(os.environ.get("SHARD_INDEX", "0"))
DEVICE = "cuda"

# Per-video discovery: any directory that DIRECTLY holds keyframe images.
# (discover_images is shallow, so this finds the real leaf video dirs no
# matter how the batches nest above them.)
video_dirs = []
frames_by_dir = {}
candidates = [kf_root, *(p for p in kf_root.rglob("*") if p.is_dir())]
for d in sorted(set(candidates)):
    imgs = discover_images(d)
    if imgs:
        video_dirs.append(d)
        frames_by_dir[d] = len(imgs)
total_frames = sum(frames_by_dir.values())
print(f"[discover] {len(video_dirs)} per-video dirs, {total_frames} keyframes "
      f"under {kf_root}", flush=True)
if not video_dirs:
    raise SystemExit("ERROR: no keyframe images found; nothing to extract")

# Strict, disjoint sharding: each shard owns the sorted videos whose index is
# congruent to SHARD_INDEX mod SHARD_COUNT. video_dirs is already sorted above,
# so the partition is identical across every shard process (no overlap, no gaps).
if shard_count > 1:
    all_dirs = video_dirs
    video_dirs = [d for i, d in enumerate(all_dirs) if i % shard_count == shard_index]
    shard_frames = sum(frames_by_dir[d] for d in video_dirs)
    print(f"[shard] {shard_index}/{shard_count}: {len(video_dirs)}/{len(all_dirs)} "
          f"videos, {shard_frames} keyframes", flush=True)
    if not video_dirs:
        raise SystemExit(f"shard {shard_index}/{shard_count} has no videos; nothing to do")

# Per-encoder batch size (H200 143 GB; qwen3vl-2B is heavier -> smaller batch).
# The qwen3vl batch is overridable via BATCH_QWEN (8B wants a smaller batch).
BATCH = {"siglip2": 256, "metaclip2": 128, "qwen3vl": int(batch_qwen) if batch_qwen else 32}


def build(name):
    if name == "siglip2":
        from aic2026.embedding.siglip2 import SigLip2Embedder

        return SigLip2Embedder(device=DEVICE)
    if name == "metaclip2":
        from aic2026.embedding.metaclip2 import MetaClip2Embedder

        return MetaClip2Embedder(device=DEVICE)
    if name == "qwen3vl":
        from aic2026.embedding.qwen3vl_embed import Qwen3VLEmbedder

        # qwen_model (HF repo id) selects the backbone; None -> Qwen3VLEmbedder
        # default (2B). The 8B lane passes Qwen/Qwen3-VL-Embedding-8B.
        kw = {"device": DEVICE, "impl_src": qwen_src}
        if qwen_model:
            kw["model_name_or_path"] = qwen_model
        return Qwen3VLEmbedder(**kw)
    raise SystemExit(f"unknown encoder {name!r}")


for enc in encoders:
    t0 = time.time()
    print(f"\n=== encoder={enc} : loading model ===", flush=True)
    emb = build(enc)
    # qwen3vl wrapper hardcodes NATIVE_DIM=2048 (the 2B repo). The 8B repo has a
    # different native dim, and extract_image_embeddings asserts vecs.shape ==
    # (batch, emb.dim). Probe the true output dim from one keyframe and override
    # emb.dim so the assertion + matrix allocation use the real width (2B: no-op).
    if enc == "qwen3vl" and video_dirs:
        probe = discover_images(video_dirs[0])[:1]
        if probe:
            real_dim = int(emb.encode_image(probe).shape[1])
            if real_dim != emb.dim:
                print(f"[qwen3vl] probed native dim={real_dim} "
                      f"(wrapper reported {emb.dim}); overriding", flush=True)
                emb.dim = real_dim
    bs = BATCH.get(enc, 32)
    dst = out_root / enc
    dst.mkdir(parents=True, exist_ok=True)
    done = skipped = nframes = 0
    n = len(video_dirs)
    for i, d in enumerate(video_dirs):
        out_base = dst / d.name
        if out_base.with_suffix(".npy").exists():
            skipped += 1
            continue
        paths = discover_images(d)
        res = extract_image_embeddings(paths, emb, out=out_base, batch_size=bs)
        done += 1
        nframes += res.n
        if done % 25 == 0 or i == n - 1:
            el = time.time() - t0
            rate = nframes / el if el > 0 else 0.0
            print(f"[{enc}] {i + 1}/{n} videos | {nframes} frames | "
                  f"{el:.0f}s | {rate:.1f} fr/s | last={d.name} dim={res.dim}",
                  flush=True)
    del emb
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"=== encoder={enc} DONE: {done} extracted, {skipped} skipped, "
          f"{nframes} frames, {time.time() - t0:.0f}s ===", flush=True)

print("\n[ALL ENCODERS DONE]", flush=True)
PY
rc=$?
echo "== extraction exit code: $rc =="
echo "== output tree =="
for e in $(printf '%s' "$ENCODERS" | tr ',' ' '); do
  npy=$(find "$OUT_ROOT/$e" -name '*.npy' 2>/dev/null | wc -l)
  man=$(find "$OUT_ROOT/$e" -name '*.manifest.jsonl' 2>/dev/null | wc -l)
  echo "   $e: $npy .npy + $man .manifest.jsonl"
done

# --- shard sentinels: tell the bank watcher this shard is done (or failed) ---
if [ "$SHARD_COUNT" -gt 1 ]; then
  mkdir -p "$SENTINEL_DIR"
  rm -f "$SENTINEL_DIR/index_shard_${SHARD_INDEX}.done" \
        "$SENTINEL_DIR/index_shard_${SHARD_INDEX}.fail" 2>/dev/null
  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) rc=0" > "$SENTINEL_DIR/index_shard_${SHARD_INDEX}.done"
    echo "== shard $SHARD_INDEX/$SHARD_COUNT sentinel: $SENTINEL_DIR/index_shard_${SHARD_INDEX}.done =="
  else
    echo "$(date -u +%FT%TZ) rc=$rc" > "$SENTINEL_DIR/index_shard_${SHARD_INDEX}.fail"
    echo "== shard $SHARD_INDEX/$SHARD_COUNT FAILED sentinel: $SENTINEL_DIR/index_shard_${SHARD_INDEX}.fail =="
  fi
fi

# --- auto-bank OUT_ROOT to R2 (mirror infra/remote/c1_bank.sh) ---
if [ "$rc" -eq 0 ] && [ "$BANK" = "1" ]; then
  ENVFILE=""
  for f in "$DIR/.env.remote" "$HOME"/aic2026/*/.env.remote "$HOME/.env.remote"; do
    [ -f "$f" ] && { ENVFILE="$f"; break; }
  done
  if [ -n "$ENVFILE" ]; then
    echo "== banking $OUT_ROOT -> R2 $R2_PREFIX/ via R2Client (env: $ENVFILE) =="
    set -a
    . "$ENVFILE" 2>/dev/null || true
    set +a
    if [ -z "${R2_BUCKET:-}" ]; then
      echo "WARN: R2_BUCKET unset after sourcing $ENVFILE; skipping auto-bank."
    else
      cd "$DIR" && OUT_ROOT="$OUT_ROOT" R2_PREFIX="$R2_PREFIX" uv run python - <<'PY'
import os
from pathlib import Path

from aic2026.remote.r2 import R2Client

client = R2Client()
prefix = os.environ["R2_PREFIX"]
keys = client.upload_dir(Path(os.environ["OUT_ROOT"]), prefix)
print(f"uploaded {len(keys)} objects under {prefix}/")
got = client.list(prefix)
print(f"verify: {len(got)} objects present under {prefix}/")
PY
    fi
  else
    echo "WARN: no .env.remote found under ~/aic2026/*/; skipping auto-bank."
    echo "      bank manually: ssh aic2026-gpu 'BANK=1 R2_PREFIX=$R2_PREFIX bash index_extract.sh' (re-run is idempotent)"
  fi
fi
echo "== index_extract finished $(date -u +%FT%TZ) rc=$rc =="
exit $rc
