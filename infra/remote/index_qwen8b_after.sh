#!/usr/bin/env bash
# Implements SPEC-0004 SS 3-4 + ADR-0012 (offline visual-document lane, 8B).
# Workstream C: dependent launcher for the Qwen3-VL-Embedding-8B offline index.
# It WAITS for the 3-encoder shards (Workstream A) to finish - so GPUs 1-6 are
# free - then runs the 8B model sharded across GPUs 1-6 exactly like A, writing
# to a DISTINCT output tree and banking to a DISTINCT R2 prefix.
#
#   scp infra/remote/index_extract.sh aic2026-gpu:index_extract_8b.sh   # probe-enabled runner
#   scp infra/remote/index_qwen8b_after.sh aic2026-gpu:.
#   ssh aic2026-gpu 'setsid nohup bash index_qwen8b_after.sh > /tmp/aic2025/index_qwen8b_launcher.log 2>&1 &'
#
# Why a distinct runner filename: A's 6 shards are still executing ~/index_extract.sh;
# overwriting a running bash script corrupts it. The probe-enabled copy is shipped
# as ~/index_extract_8b.sh and used only here. (The 8B wrapper's native dim != the
# 2B's 2048, so the runner probes the true dim at load - see index_extract.sh.)
#
# Env (all overridable):
#   RUNNER        probe-enabled runner path (default ~/index_extract_8b.sh)
#   WATCHER       bank watcher path (default ~/index_bank_watcher.sh)
#   DIR           project dir w/ embedding extra (default ~/aic2026/HEAD)
#   QWEN_MODEL    HF repo id (default Qwen/Qwen3-VL-Embedding-8B)
#   OUT_ROOT      output tree (default /tmp/aic2025/index_qwen8b)
#   R2_PREFIX     bank prefix (default index/aic2025-proxy-qwen8b-20260604)
#   SHARD_COUNT   shards = GPUs used (default 6 -> GPUs 1..6)
#   BATCH_QWEN    per-call batch (default 8; 8B is heavier than the 2B's 32)
#   WAIT_SENTINEL_DIR  where A drops index_shard_<k>.done (default /tmp/aic2025)
#   QWEN8B_SENTINEL_DIR distinct sentinel dir for the 8B shards (default /tmp/aic2025/qwen8b)
#   POLL_SECS / TIMEOUT_SECS  wait loop tuning (default 30 / 21600)

set -u
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

RUNNER="${RUNNER:-$HOME/index_extract_8b.sh}"
WATCHER="${WATCHER:-$HOME/index_bank_watcher.sh}"
DIR="${DIR:-$HOME/aic2026/HEAD}"
QWEN_SRC="${QWEN_SRC:-$HOME/Qwen3-VL-Embedding}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-Embedding-8B}"
OUT_ROOT="${OUT_ROOT:-/tmp/aic2025/index_qwen8b}"
R2_PREFIX="${R2_PREFIX:-index/aic2025-proxy-qwen8b-20260604}"
SHARD_COUNT="${SHARD_COUNT:-6}"
BATCH_QWEN="${BATCH_QWEN:-8}"
WAIT_SENTINEL_DIR="${WAIT_SENTINEL_DIR:-/tmp/aic2025}"
QWEN8B_SENTINEL_DIR="${QWEN8B_SENTINEL_DIR:-/tmp/aic2025/qwen8b}"
POLL_SECS="${POLL_SECS:-30}"
TIMEOUT_SECS="${TIMEOUT_SECS:-21600}"

[ -x "$RUNNER" ] || [ -f "$RUNNER" ] || { echo "ERROR: no runner $RUNNER"; exit 1; }
mkdir -p "$OUT_ROOT" "$QWEN8B_SENTINEL_DIR"

echo "== index_qwen8b_after start $(date -u +%FT%TZ) =="
echo "   waiting for $SHARD_COUNT 3-encoder shard sentinels in $WAIT_SENTINEL_DIR"

waited=0
while :; do
  done_n=$(find "$WAIT_SENTINEL_DIR" -maxdepth 1 -name 'index_shard_*.done' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done_n" -ge "$SHARD_COUNT" ]; then
    echo "== Workstream A done ($done_n/$SHARD_COUNT shards) at $(date -u +%FT%TZ); starting 8B =="
    break
  fi
  if [ "$waited" -ge "$TIMEOUT_SECS" ]; then
    echo "ERROR: timeout waiting for A ($done_n/$SHARD_COUNT). NOT starting 8B."
    exit 1
  fi
  sleep "$POLL_SECS"; waited=$((waited + POLL_SECS))
done

# Pre-download the 8B once so the 6 shards load from cache (avoids 6x concurrent
# ~16 GB downloads + HF cache races). Idempotent: snapshot_download is a no-op if
# already cached.
echo "== pre-fetching $QWEN_MODEL into the HF cache (once) =="
( cd "$DIR" && QWEN_MODEL="$QWEN_MODEL" uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["QWEN_MODEL"]
p = snapshot_download(repo)
print(f"cached {repo} -> {p}")
PY
) || { echo "ERROR: 8B pre-fetch failed; aborting before launching shards."; exit 1; }

echo "== launching $SHARD_COUNT 8B shards on GPUs 1..$SHARD_COUNT =="
cd "$HOME"
for k in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu=$((k + 1))
  GPU=$gpu SHARD_COUNT=$SHARD_COUNT SHARD_INDEX=$k \
    ENCODERS=qwen3vl QWEN_MODEL="$QWEN_MODEL" QWEN_SRC="$QWEN_SRC" \
    BATCH_QWEN="$BATCH_QWEN" OUT_ROOT="$OUT_ROOT" BANK=0 \
    SENTINEL_DIR="$QWEN8B_SENTINEL_DIR" \
    setsid nohup bash "$RUNNER" > "/tmp/aic2025/index_qwen8b_shard_${k}.log" 2>&1 < /dev/null &
  echo "  launched 8B shard SHARD_INDEX=$k -> GPU $gpu (launcher pid $!)"
  sleep 5
done

sleep 8
echo "== running 8B shard procs =="
pgrep -af "$RUNNER" | grep -v pgrep || echo none

echo "== launching 8B bank watcher (-> R2 $R2_PREFIX/) =="
SHARD_COUNT=$SHARD_COUNT OUT_ROOT="$OUT_ROOT" R2_PREFIX="$R2_PREFIX" \
  SENTINEL_DIR="$QWEN8B_SENTINEL_DIR" DIR="$DIR" POLL_SECS="$POLL_SECS" \
  setsid nohup bash "$WATCHER" > /tmp/aic2025/index_qwen8b_bank.log 2>&1 < /dev/null &
echo "  8B bank watcher launcher pid $!"

echo "== index_qwen8b_after finished launching $(date -u +%FT%TZ) =="
