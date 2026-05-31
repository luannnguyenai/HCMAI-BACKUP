#!/usr/bin/env bash
# C1 (SPEC-0014) ship-gate eval runner for the GPU box.
# Restores head.pt + pairs.parquet from R2 (or uses local /tmp/c1/), harvests a
# held-out query set disjoint from the training corpus, and runs the three-way
# degradation@10 comparison via `bin/train c1-eval`.
#
#   scp infra/remote/c1_eval.sh aic2026-gpu:c1_eval.sh
#   ssh -o ServerAliveInterval=30 aic2026-gpu 'bash c1_eval.sh'
#
# Optional args:
#   $1 = code-dir SHA7 (default 7f18c88 - whichever copy has bin/train installed)
#   $2 = baseline SHA7 to restore from R2 c1-baseline/<sha>/ (default = $1)
#   $3 = held-out query count (default 200)

SHA7="${1:-7f18c88}"
BASELINE_SHA="${2:-$SHA7}"
N_HELDOUT="${3:-200}"
DIR="$HOME/aic2026/$SHA7"

export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

cd "$DIR" || { echo "ERROR: no provisioned dir $DIR"; exit 1; }

# Find any .env.remote (gitignored, not in the code archive). R2Client strips a
# stray bucket suffix, so even older copies with the bad endpoint work.
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
  mkdir -p /tmp/c1_eval/baseline
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
out = Path("/tmp/c1_eval/baseline")
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
  CKPT="/tmp/c1_eval/baseline/head.pt"
  PAIRS="/tmp/c1_eval/baseline/pairs.parquet"
fi

mkdir -p /tmp/c1_eval
OUT="/tmp/c1_eval/c1_eval.json"

echo "== c1-eval (n_heldout=$N_HELDOUT, ckpt=$CKPT) =="
uv run train c1-eval \
  --checkpoint "$CKPT" \
  --build-heldout "$N_HELDOUT" \
  --exclude "$PAIRS" \
  --out "$OUT"
eval_rc=$?
echo "== c1-eval exit code: $eval_rc =="

echo "== /tmp/c1_eval =="
ls -la /tmp/c1_eval 2>/dev/null
echo "== c1_eval.json (ship-gate verdict) =="
cat "$OUT" 2>/dev/null || echo "(no $OUT)"
