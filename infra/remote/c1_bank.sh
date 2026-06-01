#!/usr/bin/env bash
# Bank the trained C1 artifacts (SPEC-0014) to R2 before the lease is reclaimed.
#   scp infra/remote/c1_bank.sh aic2026-gpu:c1_bank.sh
#   ssh aic2026-gpu 'bash c1_bank.sh [tag] [sha7]'
# Uploads /tmp/c1 (pairs.parquet + run/head.pt + run/train_meta.json) to
# s3://$R2_BUCKET/c1-baseline/<sha>[-<tag>]/ via R2Client (endpoint-guarded +
# checksum-compatible, so it works regardless of which .env.remote endpoint is
# sourced). Pass an optional `tag` to bank multiple variants side-by-side
# without overwriting (e.g. v2-ocr for the OCR-noise retrain).

TAG="${1:-}"
SHA7="${2:-7f18c88}"
DIR="$HOME/aic2026/$SHA7"
PREFIX="c1-baseline/${SHA7}${TAG:+-$TAG}"
export PATH="$HOME/.local/bin:$PATH"

# .env.remote is gitignored (not in the code archive). Find any copy for creds;
# R2Client strips a stray bucket suffix from the endpoint, so even the older
# unfixed copies work.
ENVFILE=""
for f in "$DIR/.env.remote" "$HOME"/aic2026/*/.env.remote "$HOME/.env.remote"; do
  [ -f "$f" ] && { ENVFILE="$f"; break; }
done
[ -n "$ENVFILE" ] || { echo "ERROR: no .env.remote found under ~/aic2026/*/"; exit 1; }
echo "sourcing env: $ENVFILE"
set -a; . "$ENVFILE" 2>/dev/null || true; set +a
[ -n "$R2_BUCKET" ] || { echo "ERROR: R2_BUCKET unset after sourcing $ENVFILE"; exit 1; }

echo "== train_meta.json (the result) =="
cat /tmp/c1/run/train_meta.json 2>/dev/null || { echo "(no train_meta.json)"; exit 1; }

echo "== banking /tmp/c1 -> R2 $PREFIX/ via R2Client =="
cd "$DIR" || exit 1
uv run python - "$PREFIX" <<'PY'
import sys
from pathlib import Path
from aic2026.remote.r2 import R2Client

prefix = sys.argv[1]
client = R2Client()
keys = client.upload_dir(Path("/tmp/c1"), prefix)
print(f"uploaded {len(keys)} objects:")
for k in keys:
    print("  ", k)
print("--- verify list under c1-baseline/ ---")
got = client.list("c1-baseline")
print(len(got), "objects present")
PY
