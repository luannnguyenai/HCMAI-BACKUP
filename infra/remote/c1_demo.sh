#!/usr/bin/env bash
# C1 (SPEC-0014 SS 6) live demo runner for the GPU box.
# Restores head.pt + pairs.parquet from R2 (or uses local /tmp/c1/) and runs
# `bin/train c1-demo` in canned, interactive, or both mode.
#
#   scp infra/remote/c1_demo.sh aic2026-gpu:c1_demo.sh
#   ssh -t aic2026-gpu 'bash c1_demo.sh both'        # canned + interactive REPL
#   ssh    aic2026-gpu 'bash c1_demo.sh canned'      # canned only, no REPL
#   ssh    aic2026-gpu 'bash c1_demo.sh tune'        # sweep seeds, recommend best
#
# Use `ssh -t` for interactive/both so stdin is a TTY (the REPL needs it).
# `tune` prints a per-example seed sweep + recommended noise_seed (re-tune the
# canned set when the head changes), then exits without the showcase.
#
# Optional args:
#   $1 = mode {canned, interactive, both, tune}  (default canned)
#   $2 = code-dir SHA7                      (default 7f18c88; matches c1_eval.sh)
#   $3 = baseline SHA7 to restore from R2   (default = $2)
#   $4 = n_docs (index size)                (default 2000)

MODE="${1:-canned}"
SHA7="${2:-7f18c88}"
BASELINE_SHA="${3:-$SHA7}"
N_DOCS="${4:-2000}"
DIR="$HOME/aic2026/$SHA7"

export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

cd "$DIR" || { echo "ERROR: no provisioned dir $DIR"; exit 1; }

# Find any .env.remote (gitignored, not in the code archive). Same lookup
# order as c1_eval.sh / c1_bank.sh -- DRY-ish, kept inline for portability.
ENVFILE=""
for f in "$DIR/.env.remote" "$HOME"/aic2026/*/.env.remote "$HOME/.env.remote"; do
  [ -f "$f" ] && { ENVFILE="$f"; break; }
done
[ -n "$ENVFILE" ] || { echo "ERROR: no .env.remote under ~/aic2026/*/"; exit 1; }
echo "sourcing env: $ENVFILE"
set -a; . "$ENVFILE" 2>/dev/null || true; set +a

# Prefer local /tmp/c1 when it exists (same lease as training); otherwise pull
# from R2 c1-baseline/<sha>/.
LOCAL_CKPT="/tmp/c1/run/head.pt"
LOCAL_PAIRS="/tmp/c1/pairs.parquet"
if [ -f "$LOCAL_CKPT" ] && [ -f "$LOCAL_PAIRS" ]; then
  echo "== using local baseline =="
  echo "  ckpt:  $LOCAL_CKPT"
  echo "  pairs: $LOCAL_PAIRS"
  CKPT="$LOCAL_CKPT"
  PAIRS="$LOCAL_PAIRS"
else
  echo "== restoring baseline from R2 c1-baseline/$BASELINE_SHA/ =="
  mkdir -p /tmp/c1_demo/baseline
  uv run python - "$BASELINE_SHA" <<'PY'
import sys
from pathlib import Path

from aic2026.remote.r2 import R2Client

sha = sys.argv[1]
prefix = f"c1-baseline/{sha}"
client = R2Client()
keys = client.list(prefix)
if not keys:
    raise SystemExit(f"no objects under {prefix!r} in R2")
print(f"  found {len(keys)} keys under {prefix}")
out = Path("/tmp/c1_demo/baseline")
out.mkdir(parents=True, exist_ok=True)
for suffix, name in (("head.pt", "head.pt"), ("pairs.parquet", "pairs.parquet")):
    for k in keys:
        if k.endswith(suffix):
            (out / name).write_bytes(client.get_bytes(k))
            print(f"  restored {k} -> {out / name}")
            break
    else:
        raise SystemExit(f"no {suffix} under {prefix}")
PY
  CKPT="/tmp/c1_demo/baseline/head.pt"
  PAIRS="/tmp/c1_demo/baseline/pairs.parquet"
fi

echo "== c1-demo (mode=$MODE, n_docs=$N_DOCS, ckpt=$CKPT) =="
uv run train c1-demo \
  --checkpoint "$CKPT" \
  --pairs "$PAIRS" \
  --mode "$MODE" \
  --n-docs "$N_DOCS"
rc=$?
echo "== c1-demo exit code: $rc =="
exit $rc
