#!/usr/bin/env bash
# Implements SPEC-0004 SS 3-4 + ADR-0011/ADR-0012 (banking the offline INDEX).
# Companion to the sharded index_extract.sh: waits until ALL shards have dropped
# their success sentinel (index_shard_<k>.done), then banks the whole OUT_ROOT
# tree to Cloudflare R2 ONCE under R2_PREFIX. This is how the 3-encoder (or the
# Qwen-8B) index still lands in R2 at a single prefix after a parallel run.
#
#   scp infra/remote/index_bank_watcher.sh aic2026-gpu:.
#   ssh aic2026-gpu 'SHARD_COUNT=6 OUT_ROOT=/tmp/aic2025/index \
#       R2_PREFIX=index/aic2025-proxy-3enc-20260604 \
#       setsid nohup bash index_bank_watcher.sh > /tmp/aic2025/index_bank.log 2>&1 &'
#
# Env (all overridable):
#   DIR          project dir w/ the `embedding` extra synced (default ~/aic2026/HEAD)
#   OUT_ROOT     tree to upload (default /tmp/aic2025/index)
#   R2_PREFIX    R2 destination prefix (REQUIRED in practice; default index/<ts>)
#   SHARD_COUNT  number of shard sentinels to wait for (default 6)
#   SENTINEL_DIR where the shards drop index_shard_<k>.done/.fail (default /tmp/aic2025)
#   POLL_SECS    poll interval (default 30)
#   TIMEOUT_SECS give up waiting after this many seconds (default 21600 = 6 h)

set -u
export PATH="$HOME/.local/bin:$PATH"

DIR="${DIR:-$HOME/aic2026/HEAD}"
OUT_ROOT="${OUT_ROOT:-/tmp/aic2025/index}"
R2_PREFIX="${R2_PREFIX:-index/$(date -u +%Y%m%dT%H%M%SZ)}"
SHARD_COUNT="${SHARD_COUNT:-6}"
SENTINEL_DIR="${SENTINEL_DIR:-/tmp/aic2025}"
POLL_SECS="${POLL_SECS:-30}"
TIMEOUT_SECS="${TIMEOUT_SECS:-21600}"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH"; exit 1; }
[ -d "$DIR" ] || { echo "ERROR: no project dir $DIR"; exit 1; }

echo "== index_bank_watcher start $(date -u +%FT%TZ) =="
echo "   waiting for $SHARD_COUNT shard sentinels in $SENTINEL_DIR"
echo "   OUT_ROOT=$OUT_ROOT  R2_PREFIX=$R2_PREFIX"

waited=0
while :; do
  done_n=$(find "$SENTINEL_DIR" -maxdepth 1 -name 'index_shard_*.done' 2>/dev/null | wc -l | tr -d ' ')
  fail_n=$(find "$SENTINEL_DIR" -maxdepth 1 -name 'index_shard_*.fail' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$fail_n" -gt 0 ]; then
    echo "WARN: $fail_n shard(s) reported FAILURE:"
    find "$SENTINEL_DIR" -maxdepth 1 -name 'index_shard_*.fail' -print -exec cat {} \; 2>/dev/null
    echo "      Banking will proceed only once the remaining shards finish or you"
    echo "      relaunch the failed shard(s) (idempotent: finished .npy are skipped)."
  fi
  if [ "$done_n" -ge "$SHARD_COUNT" ]; then
    echo "== all $done_n/$SHARD_COUNT shards done at $(date -u +%FT%TZ); banking =="
    break
  fi
  if [ "$waited" -ge "$TIMEOUT_SECS" ]; then
    echo "ERROR: timeout after ${TIMEOUT_SECS}s with only $done_n/$SHARD_COUNT shards done; NOT banking."
    echo "       Bank manually once shards finish:"
    echo "       ssh aic2026-gpu 'DIR=$DIR OUT_ROOT=$OUT_ROOT R2_PREFIX=$R2_PREFIX SHARD_COUNT=$SHARD_COUNT bash index_bank_watcher.sh'"
    exit 1
  fi
  sleep "$POLL_SECS"
  waited=$((waited + POLL_SECS))
done

echo "== output tree (final) =="
for e in $(ls "$OUT_ROOT" 2>/dev/null); do
  npy=$(find "$OUT_ROOT/$e" -name '*.npy' 2>/dev/null | wc -l)
  man=$(find "$OUT_ROOT/$e" -name '*.manifest.jsonl' 2>/dev/null | wc -l)
  echo "   $e: $npy .npy + $man .manifest.jsonl"
done

# --- find creds + bank OUT_ROOT to R2 (mirror index_extract.sh) ---
ENVFILE=""
for f in "$DIR/.env.remote" "$HOME"/aic2026/*/.env.remote "$HOME/.env.remote"; do
  [ -f "$f" ] && { ENVFILE="$f"; break; }
done
if [ -z "$ENVFILE" ]; then
  echo "ERROR: no .env.remote found under ~/aic2026/*/; cannot bank."
  exit 1
fi
echo "== banking $OUT_ROOT -> R2 $R2_PREFIX/ via R2Client (env: $ENVFILE) =="
set -a; . "$ENVFILE" 2>/dev/null || true; set +a
if [ -z "${R2_BUCKET:-}" ]; then
  echo "ERROR: R2_BUCKET unset after sourcing $ENVFILE; cannot bank."
  exit 1
fi

cd "$DIR" || exit 1
OUT_ROOT="$OUT_ROOT" R2_PREFIX="$R2_PREFIX" uv run python - <<'PY'
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
rc=$?
echo "== index_bank_watcher finished $(date -u +%FT%TZ) rc=$rc =="
exit $rc
