#!/usr/bin/env bash
# Implements SPEC-0014 calibration + retrain (C1 noise-schedule calibration).
# Workstream B runner, pinned to GPU 7 (strict partition: C1 never touches the
# index GPUs 1-6). End to end:
#   1. OCR a sample of keyframes (EasyOCR on GPU 7) -> real-OCR text JSONL.
#   2. Calibrate: compare real-OCR surface stats vs the committed v3 synthetic
#      noise schedule, derive nudged per-mode weights, emit a nudged corpus +
#      calibration_report.json (committed diacritic_noise.py is NOT modified).
#   3. Bank the report + nudged corpus to R2 immediately (deliverable is safe
#      even if the heavy retrain is cut off by the lease).
#   4. Retrain the C1 head on the nudged corpus, run the ship-gate eval.
#   5. Bank head.pt + train_meta.json + c1_eval.json to the same R2 prefix.
#
#   scp infra/remote/{ocr_sample.py,c1_calibrate.py,c1_calibrate_retrain.sh} aic2026-gpu:.
#   ssh aic2026-gpu 'CUDA_VISIBLE_DEVICES=7 setsid nohup bash c1_calibrate_retrain.sh \
#       > /tmp/c1cal/c1_calibrate.log 2>&1 &'
#
# Isolation: the heavy train/calibrate steps run via `uv run` in HEAD_DIR's
# already-synced venv (read-only; safe alongside the index shards). The ONLY
# venv-mutating step (installing EasyOCR) happens in the SEPARATE OCR_DIR venv so
# it can never disturb the running index extraction in HEAD_DIR.

set -u
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

HEAD_DIR="${HEAD_DIR:-$HOME/aic2026/HEAD}"      # current (v3) code + train/embedding venv
OCR_DIR="${OCR_DIR:-$HOME/aic2026/7f18c88}"     # isolated venv for the EasyOCR install
WORK="${WORK:-/tmp/c1cal}"
KF_ROOT="${KF_ROOT:-/tmp/aic2025/kf}"
N_OCR="${N_OCR:-400}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH="${BATCH:-128}"
N_HELDOUT="${N_HELDOUT:-200}"
CLEAN_FROM="${CLEAN_FROM:-/tmp/c1/pairs.parquet}"
R2_PREFIX="${R2_PREFIX:-c1/calibrated-20260604}"
SCRIPTS="${SCRIPTS:-$HOME}"  # where ocr_sample.py + c1_calibrate.py were scp'd

mkdir -p "$WORK"
echo "== c1_calibrate_retrain start $(date -u +%FT%TZ) GPU=$CUDA_VISIBLE_DEVICES =="
echo "   HEAD_DIR=$HEAD_DIR  OCR_DIR=$OCR_DIR  WORK=$WORK"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/   gpu /'

ENVFILE=""
for f in "$HEAD_DIR/.env.remote" "$HOME"/aic2026/*/.env.remote "$HOME/.env.remote"; do
  [ -f "$f" ] && { ENVFILE="$f"; break; }
done

bank() {  # bank() <prefix> <dir>
  local prefix="$1" dir="$2"
  if [ -z "$ENVFILE" ]; then echo "WARN: no .env.remote; skip bank of $dir"; return 0; fi
  set -a; . "$ENVFILE" 2>/dev/null || true; set +a
  if [ -z "${R2_BUCKET:-}" ]; then echo "WARN: R2_BUCKET unset; skip bank"; return 0; fi
  ( cd "$HEAD_DIR" && BANK_DIR="$dir" BANK_PREFIX="$prefix" uv run python - <<'PY'
import os
from pathlib import Path
from aic2026.remote.r2 import R2Client
c = R2Client()
pfx = os.environ["BANK_PREFIX"]
keys = c.upload_dir(Path(os.environ["BANK_DIR"]), pfx)
print(f"uploaded {len(keys)} objects under {pfx}/")
print(f"verify: {len(c.list(pfx))} objects present under {pfx}/")
PY
  )
}

# --- Phase 0+1: OCR sample on GPU 7 (reuse cache if present -> no venv mutation) ---
if [ -s "$WORK/ocr_sample.jsonl" ]; then
  echo "== [0+1] reusing cached OCR sample $WORK/ocr_sample.jsonl "
  echo "         (skipping EasyOCR install + OCR; zero venv mutation) =="
else
  echo "== [0] ensure EasyOCR in $OCR_DIR venv (isolated) =="
  ( cd "$OCR_DIR" && uv pip install -q easyocr 2>&1 | tail -n 3 ) || echo "WARN: easyocr install non-zero (continuing)"
  echo "== [1] OCR sample (n=$N_OCR) =="
  ( cd "$OCR_DIR" && uv run python "$SCRIPTS/ocr_sample.py" \
      --kf-root "$KF_ROOT" --n "$N_OCR" --out "$WORK/ocr_sample.jsonl" --backend easyocr --gpu 1 )
  echo "== [1] ocr exit=$? =="
  if [ ! -s "$WORK/ocr_sample.jsonl" ]; then
    echo "ERROR: no OCR output ($WORK/ocr_sample.jsonl); aborting calibration."
    exit 1
  fi
fi

# --- Phase 2: calibration report + nudged corpus (current v3 code in HEAD_DIR) ---
echo "== [2] calibrate vs v3 schedule -> report + nudged corpus =="
( cd "$HEAD_DIR" && uv run python "$SCRIPTS/c1_calibrate.py" \
    --ocr "$WORK/ocr_sample.jsonl" \
    --report "$WORK/calibration_report.json" \
    --corpus-out "$WORK/pairs_calibrated.parquet" \
    --clean-from "$CLEAN_FROM" )
cal_rc=$?
echo "== [2] calibrate exit=$cal_rc =="
if [ ! -s "$WORK/pairs_calibrated.parquet" ] || [ ! -s "$WORK/calibration_report.json" ]; then
  echo "ERROR: calibration did not produce report+corpus; aborting."
  exit 1
fi

# --- Phase 3: bank the deliverable now (report + nudged corpus) ---
echo "== [3] bank report + nudged corpus -> R2 $R2_PREFIX/ =="
bank "$R2_PREFIX" "$WORK"

# --- Phase 4: retrain the C1 head on the nudged corpus ---
echo "== [4] c1-fit on nudged corpus (max_steps=$MAX_STEPS) =="
( cd "$HEAD_DIR" && uv run train c1-fit \
    --pairs "$WORK/pairs_calibrated.parquet" \
    --out-dir "$WORK/run" \
    --max-steps "$MAX_STEPS" --batch-size "$BATCH" ) || \
  echo "NOTE: c1-fit returned non-zero (often the known at-exit GIL/tokenizers race); checking ckpt"
if [ ! -f "$WORK/run/head.pt" ]; then
  echo "ERROR: c1-fit produced no head.pt; banking calibration only."
  bank "$R2_PREFIX" "$WORK"
  exit 1
fi
echo "== [4] head.pt present =="; ls -la "$WORK/run" 2>/dev/null

# --- Phase 5: ship-gate eval on the retrained head ---
echo "== [5] c1-eval (ship-gate, n_heldout=$N_HELDOUT) =="
( cd "$HEAD_DIR" && uv run train c1-eval \
    --checkpoint "$WORK/run/head.pt" \
    --build-heldout "$N_HELDOUT" \
    --exclude "$WORK/pairs_calibrated.parquet" \
    --out "$WORK/c1_eval.json" ) || echo "NOTE: c1-eval returned non-zero"
echo "== c1_eval.json =="; cat "$WORK/c1_eval.json" 2>/dev/null || echo "(no c1_eval.json)"

# --- Phase 6: bank the retrained head + eval to the same prefix ---
echo "== [6] bank head + meta + eval -> R2 $R2_PREFIX/ =="
bank "$R2_PREFIX" "$WORK"

echo "== c1_calibrate_retrain finished $(date -u +%FT%TZ) =="
